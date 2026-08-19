/**
 * @file settings.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Simulation settings, user given 
 * @date 2023.09.12.
 * 
 * User given settings
*/

#ifndef settings_H
#define settings_H

/*
    ----------- Code settings -----------
*/

///Debug: 0-Off, 1-Low level, 2-High level
constexpr int Debug = 0;

///Use cooperative groups for GPU Wide synchronization (required for energy conservation) 0 - thread, 1 - with cooperative groups, 2 - with atomic counters
constexpr int UseGPUWideThreadSync = 1;

///Variable type used in the code (float/double)
using var_type = float;

/*
    ------------ GPU settings -----------
*/

///GPU settings
constexpr int BlockSize = 128;

/*
    ---------- Domain settings ----------
*/

///Domain settings
enum class DomainType { Rectangular, STL };
constexpr DomainType domainType = DomainType::STL;

///Save the forces acting on the triangles
constexpr bool SaveForcesTriangles = true;

/*
    -------- Particle settings ----------
*/

///Maximum number of contacts
constexpr int MaxContactNumber = 16;


/*
    -------- Solver settings ----------
*/

///Other forces
constexpr bool BodyForce = true;
constexpr bool RollingFriction = true;
constexpr bool AdhesionForce = true; //based on JKR
constexpr bool WaterBridges = true; //based on Israelachvili
constexpr var_type WaterBridgeDistanceRange = 0.006e-3f; //this is not H_o nor H_r, but the distance where water bridges are considered.

///Contact model
enum class ContactModel {Mindlin};
constexpr ContactModel contactModel = ContactModel::Mindlin;

///Contact search algorithm
enum class ContactSearch {BruteForce, DecomposedDomains, DecomposedDomainsFast, LinkedCellList };
constexpr ContactSearch contactSearch = ContactSearch::LinkedCellList;

///Time integration
enum class TimeIntegration {Euler, Exact, Adams2};
constexpr TimeIntegration timeIntegration = TimeIntegration::Euler;

///Previous accelerations stores, acceleration of particle tid is stored at tid + n*NumberOfParticles
constexpr int AccelerationStored = 1;


/*
    ----------- IO settings ------------
*/

///Time integration
enum class OutputFormat {ASCII, Binary};
constexpr OutputFormat outputFormat = OutputFormat::ASCII;

///Save settings
constexpr bool SaveVelocity = true;
constexpr bool SaveAngularVelocity = false;
constexpr bool SaveForce = true;
constexpr bool SaveTorque = false;
constexpr bool SaveId = true;
constexpr bool SaveMaterial = false;

/*
    -------- Runtime-config defaults --------
*/

///Default values used by runtime configuration parser when keys are omitted.
namespace RuntimeConfigDefaults
{
    constexpr int FixedSeed = 42690;

    constexpr var_type TimeStart = 0.0;
    constexpr var_type TimeEnd = 1.0;
    constexpr var_type Dt = 1.0e-6f;
    constexpr int SaveSteps = 100;

    constexpr var_type GravityX = 0.0;
    constexpr var_type GravityY = 0.0;
    constexpr var_type GravityZ = -9.81f;

    constexpr bool SaveTracksFirst100 = true;
}

// Domain-decomposition constants (DecomposedDomainsConstants) are defined per
// case in each example_*.cu, since mesh resolution/extent varies between runs.
// RuntimeMeshParameters lives in mesh_parameters.cuh.

#endif