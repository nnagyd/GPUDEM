/**
 * @file boundary_runtime.cuh
 * @brief Shared runtime helpers for config-driven boundary/material/layout handling.
 */

#ifndef boundary_runtime_H
#define boundary_runtime_H

#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <experimental/filesystem>

#if defined(__cpp_lib_filesystem)
namespace boundary_runtime_fs = std::filesystem;
#else
namespace boundary_runtime_fs = std::experimental::filesystem;
#endif

#include "config_runtime.cuh"
#include "domain.cuh"
#include "forces.cuh"
#include "io.cuh"
#include "material.cuh"
#include "particle.cuh"
#include "randomgen.cuh"

namespace boundaryRuntime
{
    struct BoundaryRuntimeRange
    {
        runtimeConfig::RuntimeBoundaryConfig config;
        int startId = 0;
        int endId = 0;
    };

    struct BoundaryRuntimeState
    {
        var_type velocityZ = 0.0f;
        bool trackingHeaderWritten = false;
    };

    inline bool isFinite(var_type value)
    {
        return std::isfinite(static_cast<double>(value));
    }

    inline std::string resolveBoundaryPath(const std::string &stlPath)
    {
        if (boundary_runtime_fs::exists(stlPath))
        {
            return stlPath;
        }
        const size_t slash = stlPath.find_last_of("/\\");
        const std::string fileName = (slash == std::string::npos) ? stlPath : stlPath.substr(slash + 1);
        const std::string fallback = std::string("../STL/") + fileName;
        if (boundary_runtime_fs::exists(fallback))
        {
            return fallback;
        }
        return stlPath;
    }

    inline vec3D calculateBoundaryCenter(const struct boundaryCondition &BC, int startId, int endId)
    {
        vec3D center(0.0f, 0.0f, 0.0f);
        const int count = endId - startId;
        if (count <= 0)
        {
            return center;
        }

        for (int i = startId; i < endId; ++i)
        {
            center = center + BC.p[i];
        }

        return center * (1.0f / static_cast<var_type>(count));
    }

    inline bool applyConfiguredMaterials(struct materialParameters &materials,
                                         const runtimeConfig::SimulationConfig &runtimeCfg,
                                         int maxMaterialCount,
                                         std::string &error)
    {
        if (!runtimeCfg.materials.entries.empty())
        {
            if (isFinite(runtimeCfg.materials.sigma))
            {
                materials.sigma = runtimeCfg.materials.sigma;
            }
            if (isFinite(runtimeCfg.materials.psi))
            {
                materials.psi = runtimeCfg.materials.psi;
            }
        }

        for (size_t i = 0; i < runtimeCfg.materials.entries.size(); ++i)
        {
            const runtimeConfig::RuntimeMaterialEntry &entry = runtimeCfg.materials.entries[i];
            if (!entry.isDefined)
            {
                continue;
            }

            const int materialId = (entry.id >= 0) ? entry.id : static_cast<int>(i);
            if (materialId < 0 || materialId >= maxMaterialCount)
            {
                error = "Material id out of range in runtime config: " + std::to_string(materialId);
                return false;
            }

            materials.rho[materialId] = entry.rho;
            materials.E[materialId] = entry.E;
            materials.nu[materialId] = entry.nu;
            materials.G[materialId] = entry.E / (2.0f * (1.0f + entry.nu));
            materials.e[materialId] = entry.e;
            materials.mu[materialId] = entry.mu;
            materials.mu0[materialId] = entry.mu0;
            materials.mur[materialId] = entry.mur;
            materials.theta[materialId] = entry.theta;
        }

        return true;
    }

    inline bool loadConfiguredBoundaries(struct boundaryCondition &BCsH,
                                         std::vector<BoundaryRuntimeRange> &ranges,
                                         const runtimeConfig::SimulationConfig &runtimeCfg,
                                         int maxBoundaryCount,
                                         std::string &error)
    {
        ranges.clear();
        int cursor = 0;

        for (size_t i = 0; i < runtimeCfg.boundaries.size(); ++i)
        {
            const runtimeConfig::RuntimeBoundaryConfig &cfgBoundary = runtimeCfg.boundaries[i];
            if (!cfgBoundary.enabled)
            {
                continue;
            }

            const std::string resolvedPath = resolveBoundaryPath(cfgBoundary.stlPath);
            if (!boundary_runtime_fs::exists(resolvedPath))
            {
                error = "Boundary STL not found: " + cfgBoundary.stlPath;
                return false;
            }

            if (cursor >= maxBoundaryCount)
            {
                error = "Configured boundaries exceed compile-time NumberOfBoundaries.";
                return false;
            }

            const int triangles = ioHandling::readGeometrySTL(BCsH, cursor, BoundaryConditionType::HertzWall, cfgBoundary.materialId, 1.0f, resolvedPath);
            if (triangles <= 0)
            {
                error = "Failed loading boundary STL: " + resolvedPath;
                return false;
            }

            if (cursor + triangles > maxBoundaryCount)
            {
                error = "Boundary STL triangle count exceeds compile-time NumberOfBoundaries.";
                return false;
            }

            BoundaryRuntimeRange range;
            range.config = cfgBoundary;
            range.startId = cursor;
            range.endId = cursor + triangles;
            ranges.push_back(range);

            cursor += triangles;
        }

        if (ranges.empty())
        {
            error = "No enabled boundaries were defined in runtime config.";
            return false;
        }

        return true;
    }

    inline void initializeBoundaryStates(const std::vector<BoundaryRuntimeRange> &ranges,
                                         std::vector<BoundaryRuntimeState> &states)
    {
        states.clear();
        states.resize(ranges.size());
        for (size_t i = 0; i < ranges.size(); ++i)
        {
            states[i].velocityZ = ranges[i].config.initialVelocityZ;
            states[i].trackingHeaderWritten = false;
        }
    }

    inline void applyBoundaryInitialPlacement(const std::vector<BoundaryRuntimeRange> &ranges,
                                              struct boundaryCondition &BCsH,
                                              struct boundaryCondition &BCsD)
    {
        for (size_t i = 0; i < ranges.size(); ++i)
        {
            const BoundaryRuntimeRange &range = ranges[i];
            if (range.config.mode != runtimeConfig::RuntimeBoundaryMode::FreeBodyZ)
            {
                continue;
            }

            const vec3D center = calculateBoundaryCenter(BCsH, range.startId, range.endId);

            var_type dx = 0.0f;
            var_type dy = 0.0f;
            var_type dz = 0.0f;

            if (isFinite(range.config.initialX))
            {
                dx = range.config.initialX - center.x;
            }
            if (isFinite(range.config.initialY))
            {
                dy = range.config.initialY - center.y;
            }
            if (isFinite(range.config.initialZ))
            {
                dz = range.config.initialZ - center.z;
            }

            domainHandling::translateBoundaryConditions(BCsH, range.startId, range.endId, dx, dy, dz);
            domainHandling::translateBoundaryConditions(BCsD, range.startId, range.endId, dx, dy, dz);
        }
    }

    inline var_type averageBoundaryForceZ(const struct boundaryCondition &BCsH,
                                          int startId,
                                          int endId,
                                          var_type forceAverageScale)
    {
        var_type fz = 0.0f;
        for (int i = startId; i < endId; ++i)
        {
            fz += BCsH.F[i].z;
        }
        return fz * forceAverageScale;
    }

    inline void applyBoundaryMotion(const std::vector<BoundaryRuntimeRange> &ranges,
                                    std::vector<BoundaryRuntimeState> &states,
                                    struct boundaryCondition &BCsH,
                                    struct boundaryCondition &BCsD,
                                    const struct bodyForce &gravity,
                                    var_type simulationTime,
                                    var_type saveTime,
                                    int launchIndex,
                                    var_type forceAverageScale)
    {
        for (size_t i = 0; i < ranges.size(); ++i)
        {
            const BoundaryRuntimeRange &range = ranges[i];
            const runtimeConfig::RuntimeBoundaryConfig &cfgBoundary = range.config;

            if (!cfgBoundary.enabled)
            {
                continue;
            }
            if (cfgBoundary.motion.updateIntervalLaunches <= 0)
            {
                continue;
            }
            if (((launchIndex + 1) % cfgBoundary.motion.updateIntervalLaunches) != 0)
            {
                continue;
            }
            if (simulationTime < cfgBoundary.startTime)
            {
                continue;
            }
            if (cfgBoundary.endTime >= cfgBoundary.startTime && simulationTime > cfgBoundary.endTime)
            {
                continue;
            }

            const var_type motionDt = saveTime * cfgBoundary.motion.updateIntervalLaunches;
            var_type dx = 0.0f;
            var_type dy = 0.0f;
            var_type dz = 0.0f;

            if (cfgBoundary.mode == runtimeConfig::RuntimeBoundaryMode::ForcedVelocity)
            {
                dx = cfgBoundary.velocityX * motionDt;
                dy = cfgBoundary.velocityY * motionDt;
                dz = cfgBoundary.velocityZ * motionDt;
            }
            else if (cfgBoundary.mode == runtimeConfig::RuntimeBoundaryMode::FreeBodyZ)
            {
                const var_type avgFz = averageBoundaryForceZ(BCsH, range.startId, range.endId, forceAverageScale);
                const var_type acc = gravity.z - avgFz / cfgBoundary.mass;
                states[i].velocityZ += acc * motionDt;
                dz = states[i].velocityZ * motionDt + 0.5f * acc * motionDt * motionDt;
            }
            else
            {
                continue;
            }

            domainHandling::translateBoundaryConditions(BCsH, range.startId, range.endId, dx, dy, dz);
            domainHandling::translateBoundaryConditions(BCsD, range.startId, range.endId, dx, dy, dz);
        }
    }

    inline void writeBoundaryTrackingCsv(const std::vector<BoundaryRuntimeRange> &ranges,
                                         std::vector<BoundaryRuntimeState> &states,
                                         const struct boundaryCondition &BCsH,
                                         const std::string &outputFolder,
                                         var_type simulationTime,
                                         int launchIndex)
    {
        for (size_t i = 0; i < ranges.size(); ++i)
        {
            const BoundaryRuntimeRange &range = ranges[i];
            const runtimeConfig::RuntimeBoundaryTrackingConfig &tracking = range.config.tracking;
            if (!tracking.enabled)
            {
                continue;
            }
            if (tracking.writeIntervalLaunches <= 0)
            {
                continue;
            }
            if (((launchIndex + 1) % tracking.writeIntervalLaunches) != 0)
            {
                continue;
            }

            std::string csvPath = tracking.csvPath;
            if (csvPath.empty())
            {
                csvPath = outputFolder + "/boundary_" + std::to_string(i) + "_center.csv";
            }

            if (!states[i].trackingHeaderWritten)
            {
                std::ofstream init(csvPath, std::ios::trunc);
                if (init.is_open())
                {
                    init << "time,cx,cy,cz\n";
                    init.close();
                    states[i].trackingHeaderWritten = true;
                }
            }

            std::ofstream out(csvPath, std::ios::app);
            if (!out.is_open())
            {
                continue;
            }

            const vec3D center = calculateBoundaryCenter(BCsH, range.startId, range.endId);
            out << simulationTime << "," << center.x << "," << center.y << "," << center.z << "\n";
            out.close();
        }
    }

    inline bool initializeParticlesFromLayout(struct particle &particlesH,
                                              int numberOfParticles,
                                              const runtimeConfig::SimulationConfig &runtimeCfg,
                                              std::string &error)
    {
        const runtimeConfig::RuntimeParticleLayoutConfig &layout = runtimeCfg.particleLayout;

        if (layout.mode == runtimeConfig::RuntimeParticleLayoutMode::File)
        {
            if (!boundary_runtime_fs::exists(layout.filePath))
            {
                error = "Configured particle layout file not found: " + layout.filePath;
                return false;
            }
            const int readStatus = ioHandling::readParticlesVTK(particlesH, layout.filePath, numberOfParticles);
            if (readStatus < 0)
            {
                error = "Failed reading particle layout file: " + layout.filePath;
                return false;
            }
            return true;
        }

        if (layout.mode == runtimeConfig::RuntimeParticleLayoutMode::Generated)
        {
            const runtimeConfig::RuntimeParticleGeneratedLayoutConfig &generated = layout.generated;
            RandomGeneration::initializeRandomSeed(runtimeCfg.seed);
            for (int i = 0; i < numberOfParticles; ++i)
            {
                particlesH.R[i] = generated.radius;
                particlesH.u.x[i] = RandomGeneration::randomInRange(generated.minx, generated.maxx);
                particlesH.u.y[i] = RandomGeneration::randomInRange(generated.miny, generated.maxy);
                particlesH.u.z[i] = RandomGeneration::randomInRange(generated.minz, generated.maxz);
                particlesH.v.x[i] = 0.0f;
                particlesH.v.y[i] = 0.0f;
                particlesH.v.z[i] = 0.0f;
            }
            return true;
        }

        error = "particles.layout.mode must be 'file' or 'generated'.";
        return false;
    }
}

#endif
