/**
 * @file config_runtime.cuh
 * @brief Runtime configuration data model for benchmark and scenario runs.
 */

#ifndef config_runtime_H
#define config_runtime_H

#include <limits>
#include <string>
#include <vector>

#include "settings.cuh"
#include "mesh_parameters.cuh"

namespace runtimeConfig
{
    enum class RuntimeBoundaryMode
    {
        Fixed = 0,
        ForcedVelocity = 1,
        FreeBodyZ = 2
    };

    enum class RuntimeParticleLayoutMode
    {
        Auto = 0,
        File = 1,
        Generated = 2
    };

    struct RuntimeTimeConfig
    {
        var_type start = RuntimeConfigDefaults::TimeStart;
        var_type end = RuntimeConfigDefaults::TimeEnd;
        var_type dt = RuntimeConfigDefaults::Dt;
        int saveSteps = RuntimeConfigDefaults::SaveSteps;
    };

    struct RuntimeBodyForceConfig
    {
        var_type x = RuntimeConfigDefaults::GravityX;
        var_type y = RuntimeConfigDefaults::GravityY;
        var_type z = RuntimeConfigDefaults::GravityZ;
    };

    struct RuntimeMeshConfig
    {
        int nx = DecomposedDomainsConstants::Nx;
        int ny = DecomposedDomainsConstants::Ny;
        int nz = DecomposedDomainsConstants::Nz;

        var_type minx = DecomposedDomainsConstants::minx;
        var_type miny = DecomposedDomainsConstants::miny;
        var_type minz = DecomposedDomainsConstants::minz;
        var_type maxx = DecomposedDomainsConstants::maxx;
        var_type maxy = DecomposedDomainsConstants::maxy;
        var_type maxz = DecomposedDomainsConstants::maxz;

        bool meshBoundsRuntimeApplied = false;
        bool meshCountsRuntimeApplied = false;

        RuntimeMeshParameters toRuntimeMeshParameters() const
        {
            RuntimeMeshParameters mesh{};
            mesh.nx = nx;
            mesh.ny = ny;
            mesh.nz = nz;
            mesh.ncell = nx * ny * nz;

            mesh.minx = minx;
            mesh.miny = miny;
            mesh.minz = minz;
            mesh.maxx = maxx;
            mesh.maxy = maxy;
            mesh.maxz = maxz;

            mesh.NoverDx = var_type(mesh.nx) / (mesh.maxx - mesh.minx);
            mesh.NoverDy = var_type(mesh.ny) / (mesh.maxy - mesh.miny);
            mesh.NoverDz = var_type(mesh.nz) / (mesh.maxz - mesh.minz);
            return mesh;
        }
    };

    struct RuntimeOutputConfig
    {
        bool saveVelocity = SaveVelocity;
        bool saveAngularVelocity = SaveAngularVelocity;
        bool saveForce = SaveForce;
        bool saveTorque = SaveTorque;
        bool saveId = SaveId;
        bool saveMaterial = SaveMaterial;
        bool saveTracksFirst100 = RuntimeConfigDefaults::SaveTracksFirst100;
    };

    struct RuntimeBoundaryMotionConfig
    {
        int updateIntervalLaunches = 1;
    };

    struct RuntimeMaterialEntry
    {
        bool isDefined = false;
        int id = -1;
        bool hasRho = false;
        bool hasE = false;
        bool hasNu = false;
        bool hasErest = false;
        bool hasMu = false;
        bool hasMu0 = false;
        bool hasMur = false;
        bool hasTheta = false;
        var_type rho = std::numeric_limits<var_type>::quiet_NaN();
        var_type E = std::numeric_limits<var_type>::quiet_NaN();
        var_type nu = std::numeric_limits<var_type>::quiet_NaN();
        var_type e = std::numeric_limits<var_type>::quiet_NaN();
        var_type mu = std::numeric_limits<var_type>::quiet_NaN();
        var_type mu0 = std::numeric_limits<var_type>::quiet_NaN();
        var_type mur = std::numeric_limits<var_type>::quiet_NaN();
        var_type theta = std::numeric_limits<var_type>::quiet_NaN();
    };

    struct RuntimeMaterialConfig
    {
        bool hasSigma = false;
        bool hasPsi = false;
        var_type sigma = std::numeric_limits<var_type>::quiet_NaN();
        var_type psi = std::numeric_limits<var_type>::quiet_NaN();
        std::vector<RuntimeMaterialEntry> entries;
    };

    struct RuntimeParticleGeneratedLayoutConfig
    {
        var_type radius = std::numeric_limits<var_type>::quiet_NaN();
        var_type minx = std::numeric_limits<var_type>::quiet_NaN();
        var_type maxx = std::numeric_limits<var_type>::quiet_NaN();
        var_type miny = std::numeric_limits<var_type>::quiet_NaN();
        var_type maxy = std::numeric_limits<var_type>::quiet_NaN();
        var_type minz = std::numeric_limits<var_type>::quiet_NaN();
        var_type maxz = std::numeric_limits<var_type>::quiet_NaN();
    };

    struct RuntimeParticleLayoutConfig
    {
        bool hasMode = false;
        RuntimeParticleLayoutMode mode = RuntimeParticleLayoutMode::Auto;
        std::string filePath;
        RuntimeParticleGeneratedLayoutConfig generated;
    };

    struct RuntimeBoundaryTrackingConfig
    {
        bool enabled = false;
        std::string csvPath;
        int writeIntervalLaunches = 1;
    };

    struct RuntimeBoundaryConfig
    {
        bool hasEnabled = false;
        bool hasStlPath = false;
        bool hasMaterial = false;
        bool hasMode = false;

        bool enabled = true;
        std::string name;
        std::string stlPath;
        int materialId = -1;
        RuntimeBoundaryMode mode = RuntimeBoundaryMode::Fixed;

        var_type velocityX = std::numeric_limits<var_type>::quiet_NaN();
        var_type velocityY = std::numeric_limits<var_type>::quiet_NaN();
        var_type velocityZ = std::numeric_limits<var_type>::quiet_NaN();

        var_type mass = std::numeric_limits<var_type>::quiet_NaN();
        var_type initialX = std::numeric_limits<var_type>::quiet_NaN();
        var_type initialY = std::numeric_limits<var_type>::quiet_NaN();
        var_type initialZ = std::numeric_limits<var_type>::quiet_NaN();
        var_type initialVelocityZ = 0.0f;

        var_type startTime = 0.0f;
        var_type endTime = -1.0f;

        RuntimeBoundaryMotionConfig motion;
        RuntimeBoundaryTrackingConfig tracking;
    };

    struct SimulationConfig
    {
        bool isLoadedFromFile = false;
        std::string sourceFile;

        int seed = RuntimeConfigDefaults::FixedSeed;
        RuntimeTimeConfig time;
        RuntimeBodyForceConfig gravity;
        RuntimeMeshConfig mesh;
        RuntimeOutputConfig output;
        RuntimeMaterialConfig materials;
        RuntimeParticleLayoutConfig particleLayout;
        std::vector<RuntimeBoundaryConfig> boundaries;
    };
}

#endif
