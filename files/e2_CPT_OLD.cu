#include <chrono>
#include <experimental/filesystem>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#if defined(__cpp_lib_filesystem)
namespace fs = std::filesystem;
#else
namespace fs = std::experimental::filesystem;
#endif

#include "source/settings.cuh"

#ifndef EXAMPLE_PARTICLES
#define EXAMPLE_PARTICLES 16000
#endif

constexpr int NumberOfParticles = EXAMPLE_PARTICLES;
int NumberOfActiveParticles = NumberOfParticles;
constexpr int NumberOfMaterials = 3;
constexpr int NumberOfBoundaries = 362;

namespace DecomposedDomainsConstants
{
    constexpr int Dimension = 3;
    constexpr int NpCellMax = 64;

    constexpr int Nx = 32;
    constexpr int Ny = 32;
    constexpr int Nz = 128;
    constexpr int Ncell = Nx * Ny * Nz;

    constexpr var_type minx = -0.25;
    constexpr var_type miny = -0.25;
    constexpr var_type minz = 0.0;
    constexpr var_type maxx = 0.25;
    constexpr var_type maxy = 0.25;
    constexpr var_type maxz = 2.0;

    constexpr var_type NoverDx = var_type(Nx) / (maxx - minx);
    constexpr var_type NoverDy = var_type(Ny) / (maxy - miny);
    constexpr var_type NoverDz = var_type(Nz) / (maxz - minz);
}

#include "source/solver.cuh"
#include "source/boundary_runtime.cuh"

int main(int argc, char const *argv[])
{
    runtimeConfig::SimulationConfig runtimeCfg;
    if (argc > 1)
    {
        std::string runtimeCfgError;
        if (!ioHandling::loadSimulationConfig(argv[1], runtimeCfg, runtimeCfgError))
        {
            std::cerr << runtimeCfgError << "\n";
            return -2;
        }
        RandomGeneration::initializeRandomSeed(runtimeCfg.seed);
        std::cout << "Loaded runtime config from " << argv[1] << "\n";
    }

    if (runtimeCfg.boundaries.empty())
    {
        std::cerr << "Unified config required: boundaries.<i> entries are missing.\n";
        return -3;
    }

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);
    printf("Max Grid Size: %d, %d, %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);

    struct materialParameters materials{};

    std::string materialError;
    if (!boundaryRuntime::applyConfiguredMaterials(materials, runtimeCfg, NumberOfMaterials, materialError))
    {
        std::cerr << materialError << "\n";
        return -4;
    }
    materialHandling::calculateMaterialContact(materials, materialHandling::methods::Max, materialHandling::methods::HarmonicMean, materialHandling::methods::HarmonicMean);
    materialHandling::printMaterialInfo(materials, true);

    struct timestepping timestep(runtimeCfg.time.start, runtimeCfg.time.end, runtimeCfg.time.dt, runtimeCfg.time.saveSteps);

    struct bodyForce gravity;
    gravity.x = runtimeCfg.gravity.x;
    gravity.y = runtimeCfg.gravity.y;
    gravity.z = runtimeCfg.gravity.z;

    struct boundaryCondition BCsH;
    struct boundaryCondition BCsD;
    std::vector<boundaryRuntime::BoundaryRuntimeRange> boundaryRanges;
    std::vector<boundaryRuntime::BoundaryRuntimeState> boundaryStates;

    std::string boundaryError;
    if (!boundaryRuntime::loadConfiguredBoundaries(BCsH, boundaryRanges, runtimeCfg, NumberOfBoundaries, boundaryError))
    {
        std::cerr << boundaryError << "\n";
        return -5;
    }

    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);

    std::string layoutError;
    if (!boundaryRuntime::initializeParticlesFromLayout(particlesH, NumberOfParticles, runtimeCfg, layoutError))
    {
        std::cerr << layoutError << "\n";
        return -6;
    }

    particleHandling::generateParticleParameters(particlesH, materials, 0, 0, NumberOfParticles);
    NumberOfActiveParticles = NumberOfParticles;

    struct particle particlesD;
    memoryHandling::allocateDeviceParticles(particlesD);
    memoryHandling::synchronizeParticles(particlesD, particlesH, memoryHandling::listOfVariables::All, cudaMemcpyHostToDevice);

    domainHandling::convertBoundaryConditions(BCsH, BCsD);
    memoryHandling::allocateDeviceBoundary(BCsH, BCsD);
    boundaryCondition *dBoundaryDescriptor = allocateBoundaryDescriptor();
    boundaryRuntime::initializeBoundaryStates(boundaryRanges, boundaryStates);
    boundaryRuntime::applyBoundaryInitialPlacement(boundaryRanges, BCsH, BCsD);

    std::string outputFolder = "output";
    if (fs::exists(outputFolder))
    {
        fs::remove_all(outputFolder);
    }
    fs::create_directory(outputFolder);
    ioHandling::saveEffectiveConfig(outputFolder + "/effective_runtime_config.txt", runtimeCfg);

    std::ofstream energy(outputFolder + "/energy.csv");
    std::ofstream forces(outputFolder + "/forces.csv");
    energy << "t\tKin.\tPot.\tTot.\n";
    forces << "t,Fz\n";

    int *d_syncCounter;
    cudaMalloc(&d_syncCounter, sizeof(int) * timestep.saveSteps * 3);
    cudaMemset(d_syncCounter, 0, sizeof(int) * timestep.saveSteps * 3);

    int GridSize = (NumberOfActiveParticles + BlockSize - 1) / BlockSize;
    std::cout << "<<<" << GridSize << "," << BlockSize << ">>>\n";
    int numberOfLaunches = (timestep.numberOfSteps + 1) / timestep.saveSteps;
    const RuntimeMeshParameters runtimeMesh = runtimeCfg.mesh.toRuntimeMeshParameters();
    setRuntimeMeshParameters(runtimeMesh);

    float simulationTime = 0.0f;
    auto startTime = std::chrono::steady_clock::now();
    for (int i = 0; i < numberOfLaunches; i++)
    {
        synchronizeBoundaryDescriptor(dBoundaryDescriptor, BCsD);
        void *kernelArgs[] = {
            (void *)&particlesD,
            (void *)&NumberOfActiveParticles,
            (void *)&materials,
            (void *)&timestep,
            (void *)&gravity,
            (void *)&dBoundaryDescriptor,
            (void *)&i,
            &d_syncCounter,
            &GridSize};
        cudaLaunchCooperativeKernel((void *)solver, GridSize, BlockSize, kernelArgs);
        CHECK(cudaDeviceSynchronize());

        memoryHandling::synchronizeBoundary(BCsH, BCsD);
        simulationTime += timestep.savetime;

        boundaryRuntime::applyBoundaryMotion(
            boundaryRanges,
            boundaryStates,
            BCsH,
            BCsD,
            gravity,
            simulationTime,
            timestep.savetime,
            i,
            1.0f / timestep.saveSteps);

        boundaryRuntime::writeBoundaryTrackingCsv(
            boundaryRanges,
            boundaryStates,
            BCsH,
            outputFolder,
            simulationTime,
            i);

        vec3D Fsum = vec3D(0.0f, 0.0f, 0.0f);
        float coneZ = 0.0f;
        for (size_t b = 0; b < boundaryRanges.size(); ++b)
        {
            if (boundaryRanges[b].config.mode == runtimeConfig::RuntimeBoundaryMode::Fixed)
            {
                continue;
            }
            for (int j = boundaryRanges[b].startId; j < boundaryRanges[b].endId; ++j)
            {
                Fsum = Fsum + BCsH.F[j];
            }
            coneZ = boundaryRuntime::calculateBoundaryCenter(BCsH, boundaryRanges[b].startId, boundaryRanges[b].endId).z;
            break;
        }
        Fsum = Fsum * (1.0f / timestep.saveSteps);

        if (i % 10 == 0)
        {
            memoryHandling::synchronizeParticles(particlesH, particlesD, memoryHandling::listOfVariables::Position, cudaMemcpyDeviceToHost);
            memoryHandling::synchronizeParticles(particlesH, particlesD, memoryHandling::listOfVariables::Velocity, cudaMemcpyDeviceToHost);
            float K = forceHandling::calculateTotalKineticEnergy(particlesH, NumberOfActiveParticles);
            float P = forceHandling::calculateTotalPotentialEnergy(particlesH, gravity, NumberOfActiveParticles);
            energy << simulationTime << "\t" << K << "\t" << P << "\t" << K + P << "\n";
            std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\t t = " << simulationTime << "\n";
            std::cout << "\tz = " << coneZ << "\tFz = " << Fsum.z << "\n";
            std::string name = outputFolder + "/particles_" + std::to_string(i) + ".vtu";
            ioHandling::saveParticlesVTK(NumberOfActiveParticles, particlesH, name);
        }

        forces << simulationTime << "," << Fsum.z << "\n";

        for (int j = 0; j < NumberOfBoundaries; j++)
        {
            BCsH.F[j].x = 0.0f;
            BCsH.F[j].y = 0.0f;
            BCsH.F[j].z = 0.0f;
        }

        cudaMemcpy(BCsD.F, BCsH.F, sizeof(vec3D) * NumberOfBoundaries, cudaMemcpyHostToDevice);
        cudaMemset(d_syncCounter, 0, sizeof(int) * timestep.saveSteps * 3);
    }

    auto endTime = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "Runtime: " << duration / 1000 << " ms" << std::endl;

    energy.close();
    forces.close();

    cudaFree(d_syncCounter);
    freeBoundaryDescriptor(dBoundaryDescriptor);
    memoryHandling::freeHostParticles(particlesH);
    memoryHandling::freeDeviceParticles(particlesD);
    return 0;
}
