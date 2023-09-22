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

///Use cooperative groups for GPU Wide synchronization (required for energy conservation)
constexpr bool UseGPUWideThreadSync = false;

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
constexpr int MaxContactNumber = 10;


/*
    -------- Solver settings ----------
*/

///Body forces (gravity)
constexpr bool BodyForce = true;
constexpr bool RollingFriction = true;

///Contact model
enum class ContactModel {Mindlin};
constexpr ContactModel contactModel = ContactModel::Mindlin;

///Contact search algorithm
enum class ContactSearch {BruteForce, DecomposedDomains, DecomposedDomainsFast };
constexpr ContactSearch contactSearch = ContactSearch::DecomposedDomains;

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
constexpr bool SaveForce = false;
constexpr bool SaveTorque = false;
constexpr bool SaveId = true;
constexpr bool SaveMaterial = false;

/*
    -------- Setting specifics ------------
*/

/* \brief Settings of the decomposed domains algorithm
 * 
*/
namespace DecomposedDomainsConstants
{
    ///Dimension of mesh (1,2,3)
    constexpr int Dimension = 1;

    ///Number of cell in x,y,z direction
    constexpr int Nx = 120;
    constexpr int Ny = 100;
    constexpr int Nz = 50;

    ///Min of coordinates
    constexpr var_type minx = -0.5;
    constexpr var_type miny = -0.3;
    constexpr var_type minz = 0.0;

    ///Max of coordinates
    constexpr var_type maxx = 0.5;
    constexpr var_type maxy = 0.3;
    constexpr var_type maxz = 0.2;

    ///DO NOT MODIFY - 1/max-min pre-calculated
    constexpr var_type NoverDx = var_type(Nx)/(maxx-minx);
    constexpr var_type NoverDy = var_type(Ny)/(maxy-miny);
    constexpr var_type NoverDz = var_type(Nz)/(maxz-minz);
}

#endif