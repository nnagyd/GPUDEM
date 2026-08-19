# GPUDEM — How-To Guide

A practical guide to building, running, and understanding the GPUDEM discrete
element solver. For the exhaustive list of config keys see
[RUNTIME_CONFIG_REFERENCE.md](RUNTIME_CONFIG_REFERENCE.md); for a per-example
walkthrough see [EXAMPLES.md](EXAMPLES.md).

---

## 1. What the software does

GPUDEM is a GPU-accelerated **Discrete Element Method (DEM)** solver. It tracks a
large number of spherical particles and integrates their motion under gravity and
pairwise contact forces, with rigid STL wall/boundary geometry. Everything that
changes from case to case (time stepping, mesh, materials, particle layout,
boundaries) is driven from a plain-text `.txt` config file, so a single compiled
binary runs any scenario.

Each particle carries: position $\mathbf{u}$, translational velocity $\mathbf{v}$,
angular velocity $\boldsymbol\omega$, radius $R$, mass $m$, and moment of inertia
$\theta$. Boundaries are triangulated surfaces (STL) that act as Hertzian walls.

---

## 2. Requirements

- NVIDIA GPU with CUDA support
- CUDA toolkit (`nvcc` on `PATH`, e.g. `export PATH=/usr/local/cuda/bin:$PATH`)
- A C++14 compiler with `std::experimental::filesystem` (linked via `-lstdc++fs`

The build flags live in [files/makefile](../files/makefile):

```
-O3 -std=c++14 --allow-unsupported-compiler -lineinfo -maxrregcount=32 \
--ptxas-options=-v --use_fast_math --gpu-architecture=sm_86 -I../ -lstdc++fs
```

If your GPU is a different architecture, change `--gpu-architecture=sm_XX`.

---

## 3. Building and running

All commands run from the `files/` directory.

```bash
cd files
export PATH=/usr/local/cuda/bin:$PATH   # if nvcc is not already on PATH

make example_1            # compile scenario 1 -> ./GPUDEM
./GPUDEM example_1.txt    # run it with a config
```

The binary always takes **one argument**: the path to a runtime config file.

| Target            | Produces                    | Notes                                   |
| ----------------- | --------------------------- | --------------------------------------- |
| `make e1`  | `GPUDEM` (deposition)       | Free-body loading plate                 |
| `make e2`  | `GPUDEM` (cone penetration) | Forced cone into a bed                  |
| `make examples`   | builds 1, 2              |                                         |
| `make clean`      | removes `GPUDEM`, `output/` |                                         |


```bash
make e1 && ./GPUDEM e1_deposition.txt   # ~0.1 s smoke test
```

### What a run prints

The solver prints the device name, the material/pairing table, the launch
configuration `<<<GridSize,BlockSize>>>`, and the total wall-clock `Runtime`.

---

## 4. Configuration file format

Configs are `key=value` lines; `#` starts a comment. Values are grouped by prefix.
A minimal mental model:

```ini
time.start / time.end / time.dt / time.save_steps   # time stepping
gravity.x / gravity.y / gravity.z                   # body force
mesh.min* / mesh.max* / mesh.n*                      # linked-cell grid (runtime)
output.save_*                                        # which fields to write
materials.sigma / materials.psi                     # surface tension, liquid ratio
materials.<i>.rho/E/nu/e/mu/mu0/mur/theta            # per-material properties
particles.layout.mode = generated | file            # how particles start
boundaries.<i>.*                                     # STL geometry + motion
```

Everything is validated at load time; a bad or missing key aborts the run with a
descriptive message rather than producing garbage. See
[RUNTIME_CONFIG_REFERENCE.md](RUNTIME_CONFIG_REFERENCE.md) for the full key list.

### Particle layout

- `mode=generated` — a random cloud is created inside the given box
  (`particles.layout.generated.{radius,minx,maxx,miny,maxy,minz,maxz}`) using the
  fixed `seed`, so runs are reproducible.
- `mode=file` — particles are read from a VTU file
  (`particles.layout.file_path`).

### Boundary modes

Each `boundaries.<i>` references an STL file and a material, and picks a motion mode:

| Mode           | Behaviour                                                                 |
| -------------- | ------------------------------------------------------------------------- |
| `fixed`        | Static geometry (container, walls).                                        |
| `forced`       | Moves at a prescribed constant velocity `velocity.{x,y,z}`.               |
| `free_body_z`  | Rigid body free to move in $z$ under net contact force and gravity; needs `mass` and `initial.{x,y,z}`. Models a loading plate. |

`start_time` / `end_time` gate when motion is active (`end_time=-1` means "no
end"). `motion_update_interval` sets how often (in solver launches) the boundary
is advanced. `tracking.enabled=1` writes the boundary centre to a CSV.

---

## 5. Output files

A run wipes and recreates the `output/` folder and writes:

- `output/effective_runtime_config.txt` — the fully-resolved config actually used
  (handy for reproducing a run).
- `output/particles_<i>.vtu` — particle snapshots (open in ParaView).
- `output/boundary_<b>_<i>.stl` — boundary geometry snapshots (example 1).
- `output/energy.csv` — kinetic / potential / total energy over time.
- `output/forces.csv` — scenario-specific boundary force diagnostics (examples 2, 3).
- `output/boundary_<b>_center.csv` — tracked boundary centre position (when enabled).

---

## 6. Compile-time vs runtime settings

Two settings layers exist:

- **Runtime (`.txt`)** — physical case definition: time, gravity, mesh extent/
  resolution, materials, particle cloud, boundaries. Change freely, no rebuild.
- **Compile-time** — algorithmic choices and static array sizes that must be known
  by the GPU kernels:
  - [source/settings.cuh](../source/settings.cuh): variable precision
    (`var_type = float`), block size, contact model, **contact-search algorithm**,
    **time-integration scheme**, and which physical effects are enabled
    (rolling friction, adhesion, water bridges).
  - Per-example in each `example_N.cu`: `NumberOfParticles`, `NumberOfMaterials`,
    `NumberOfBoundaries`, and `DecomposedDomainsConstants` (the linked-cell grid
    maxima `Nx,Ny,Nz,NpCellMax` and default extent). These size static device
    buffers, so they live with the case, not in global settings.

The runtime `mesh.*` keys override the mesh at run time but must stay within the
compile-time `DecomposedDomainsConstants` maxima (this is checked at load).

---

## 7. The physics

GPUDEM uses a **soft-sphere DEM**: particles are allowed to overlap slightly, and
the overlap drives a repulsive contact force. The simulation loop each substep is:

1. Build the neighbour structure (contact search).
2. For every particle, find contacts and compute contact forces/torques.
3. Add gravity, integrate velocity and position.
4. Synchronise the whole GPU grid and advance to the next substep.

### 7.1 Contact search

Selectable in `settings.cuh` via `contactSearch`:

- **LinkedCellList** (default) — the domain is divided into an $N_x\times N_y\times
  N_z$ grid. Each particle is binned into a cell; neighbours are only sought in the
  27 neighbouring cells. This makes neighbour search $O(N)$ instead of the $O(N^2)$
  brute-force cost. `NpCellMax` caps particles per cell.
- **DecomposedDomains / DecomposedDomainsFast** — cell-decomposition variants.
- **BruteForce** — all-pairs, for validation on small systems.

Up to `MaxContactNumber` (16) simultaneous contacts are stored per particle.

### 7.2 Contact kinematics

For two spheres $i,j$ with centre distance $d$ and radii $R_i,R_j$, the normal
overlap is

$$\delta_n = (R_i + R_j) - d .$$

$\delta_n>0$ means they are in contact. The unit normal $\mathbf{n}$ points between
centres, and the contact point sits at the midpoint. Effective radius and mass:

$$R^\* = \left(\tfrac{1}{R_i}+\tfrac{1}{R_j}\right)^{-1},\qquad
m^\* = \left(\tfrac{1}{m_i}+\tfrac{1}{m_j}\right)^{-1}.$$

The **tangential overlap** $\boldsymbol{\delta_t}$ is accumulated over time
($\boldsymbol{\delta_t} \mathrel{+}= \mathbf{v}_{t,\text{rel}}\,\Delta t$) and reset
when the contact breaks — this gives the model its history dependence.

### 7.3 Contact force — Hertz–Mindlin

With $R_\delta = \sqrt{R^\*\,|\delta_n|}$ and equivalent moduli $E^\*,G^\*$:

- **Normal elastic (Hertz):**
  $$F_{ne} = \tfrac{4}{3}\,E^\*\,R_\delta\,\delta_n .$$
- **Normal / tangential stiffness:** $S_n = 2\,E^\*R_\delta$, $S_t = 8\,G^\*R_\delta$.
- **Tangential elastic (Mindlin):** $\mathbf{F}_{te} = -S_t\,\boldsymbol{\delta_t}$.
- **Damping** (normal and tangential), scaled by a restitution-derived factor
  $\beta^\*$: $F_{nd}\propto \beta^\*\sqrt{S_n m^\*}\,\mathbf{v}_{n,\text{rel}}$ and
  similarly for $F_{td}$.
- **Coulomb friction cap:** if $|\mathbf{F}_t| > \mu_0^\*\,|\mathbf{F}_n|$ the contact
  slides and $\mathbf{F}_t$ is rescaled to $\mu^\*|\mathbf{F}_n|$.
- **Torque:** $\mathbf{M} = \mathbf{p}\times\mathbf{F}_t$.
- **Rolling friction** (if enabled): an opposing moment
  $\mathbf{M}_r = -\mu_r^\*|\mathbf{F}_n|\,|\mathbf{p}|\,\hat{\boldsymbol\omega}$.

### 7.4 Cohesion / adhesion (JKR)

When `AdhesionForce` is enabled, a JKR-type attractive term based on surface energy
$\sigma$ (`materials.sigma`) reduces the normal force:

$$F_{ne} \mathrel{-}= \sqrt{16\pi\,\sigma\,E^\*R_\delta}\;R_\delta .$$

This lets particles stick, forming a pull-off (adhesive) force at separation.

### 7.5 Capillary water bridges (Israelachvili)

When `WaterBridges` is enabled and two particles are *just* separated
($\delta_n<0$) within range, a liquid bridge produces a capillary attraction. The
bridge liquid volume follows from `materials.psi` (liquid volume per unit surface
area), and the force uses surface tension $\sigma$ and the pair contact angle
$\theta^\*$. Beyond the bridge rupture distance the force drops to zero.

### 7.6 Equivalent (paired) material properties

`calculateMaterialContact` precomputes per-pair reduced properties:

- **Reduced elastic modulus** (HarmonicMean, the physical Hertz combination):
  $$\frac{1}{E^\*} = \frac{1-\nu_i^2}{E_i} + \frac{1-\nu_j^2}{E_j},\qquad
    G^\* = \tfrac{1}{2}\,\frac{E^\*}{1+\nu^\*}.$$
- **Shear modulus per material** from the config: $G = \dfrac{E}{2(1+\nu)}$.
- **Damping from restitution** $e$:
  $$\beta = \frac{-\ln e}{\sqrt{\ln^2 e + \pi^2}} .$$
- **Friction coefficients** ($\mu,\mu_0,\mu_r$) combined via `Min`, `Max`, or
  `Mean` — chosen per scenario (examples 1 & 3 use `Max`, example 2 uses `Min`).
- **Contact angle:** $\theta^\* = \arccos\!\big(\tfrac12(\cos\theta_i+\cos\theta_j)\big)$.

### 7.7 Time integration

Selectable via `timeIntegration` in `settings.cuh`. Accelerations are
$\mathbf{a}=\mathbf{F}/m + \mathbf{g}$ and $\boldsymbol\beta=\mathbf{M}/\theta$.

- **Euler (RK1, default):** $\mathbf{v}\mathrel{+}=\mathbf{a}\,\Delta t$,
  $\mathbf{u}\mathrel{+}=\mathbf{v}\,\Delta t$.
- **Exact:** same velocity update, position with the $\tfrac12\mathbf{a}\Delta t^2$
  term.
- **Adams2:** 2nd-order Adams–Bashforth using the previous acceleration.

Because DEM is explicit, `time.dt` must be well below the contact (Rayleigh) time
step for stability — the examples use $\Delta t = 10^{-6}\,\text{s}$.

### 7.8 GPU execution model

One CUDA thread integrates one particle. With `UseGPUWideThreadSync=1` the kernel
uses **cooperative groups** for a grid-wide barrier every substep, which keeps all
particles on the same time level (needed for energy conservation). The kernel runs
`time.save_steps` substeps per launch; between launches the host copies data back,
advances boundaries, and writes output.

### 7.9 Boundaries and force feedback

Boundary STL triangles interact with particles through the same Hertz–Mindlin law.
The contact forces on a boundary's triangles are accumulated across a launch and
averaged. For a `free_body_z` boundary this drives its rigid-body motion:

$$a_z = g_z - \frac{\langle F_z\rangle}{m_\text{plate}},$$

integrated once per motion update. `forced` boundaries ignore forces and translate
at their set velocity. These averaged forces are also what `forces.csv` reports.

### 7.10 Energy diagnostics

$$K = \sum_i \tfrac12 m_i v_i^2 + \tfrac12 \theta_i \omega_i^2,\qquad
  P = -\sum_i m_i\,(\mathbf{u}_i\cdot\mathbf{g}).$$

Both are written to `energy.csv`; for a conservative setup $K+P$ should be
well-behaved, which is a good sanity check on `dt` and stiffness.

---

## 8. Making your own case

1. Copy the closest `example_N.txt` to `mycase.txt` and edit materials, particle
   box, gravity, mesh, and boundaries.
2. Make sure every referenced STL exists in `files/` or `STL/`.
3. Keep `mesh.n{x,y,z}` within the example's `DecomposedDomainsConstants` maxima
   (raise them in the `.cu` and rebuild if you need a finer grid).
4. If the particle count changes, update `NumberOfParticles` (or use the
   `-DEXAMPLE_PARTICLES=` override) and `NumberOfBoundaries` to match the STL
   triangle total, then rebuild.
5. Run `./GPUDEM mycase.txt` and inspect `output/`.
