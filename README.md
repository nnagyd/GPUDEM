# GPUDEM New

GPUDEM New is a CUDA-based Discrete Element Method solver package prepared for reproducible research workflows.

This project includes:
- Runtime-configurable simulation controls (time, gravity, mesh)
- Runtime-configurable moving STL boundary motion controls
- Example configurations for deposition, shear box testing, and cone penetration

The code is provided under the GNU General Public License (see LICENSE).

## Installation

Prerequisites:
- CUDA toolkit compatible with your GPU architecture
- A compatible host compiler

Build from the files directory:

```bash
cd files
make e1
```

## Documentation

See docs for publication-oriented material:
- HOWTO.md — build/run guide, config format, and the physics solved
