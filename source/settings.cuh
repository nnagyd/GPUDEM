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
constexpr bool UseGPUWideThreadSync = true;

///Variable type used in the code (float/double)
using var_type = float;

/*
    ------------ GPU settings -----------
*/

///GPU settings
constexpr int BlockSize = 64;

/*
    ---------- Domain settings ----------
*/

///Domain settings
enum class DomainType { Rectangular, STL };
constexpr DomainType domainType = DomainType::STL;

/*
    -------- Particle settings ----------
*/

///Maximum number of contacts
constexpr int MaxContactNumber = 16;


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
enum class ContactSearch {BruteForce, DecomposedDomains, DecomposedDomainsFast, Balanced};
constexpr ContactSearch contactSearch = ContactSearch::BruteForce;

///Time integration
enum class TimeIntegration {Euler, Exact, Adams2};
constexpr TimeIntegration timeIntegration = TimeIntegration::Exact;

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
constexpr bool SaveAngularVelocity = true;
constexpr bool SaveForce = false;
constexpr bool SaveTorque = false;
constexpr bool SaveId = false;
constexpr bool SaveMaterial = true;

/*
    -------- Setting specifics ------------
*/

/* \brief Settings of the decomposed domains algorithm
 * 
*/
namespace DecomposedDomainsConstants
{
    ///Number of cell in x,y,z direction
    constexpr int Nx = 100;
    constexpr int Ny = 100;
    constexpr int Nz = 200;

    ///Min of coordinates
    constexpr var_type minx = -1.0;
    constexpr var_type miny = -1.0;
    constexpr var_type minz = 0.0;

    ///Max of coordinates
    constexpr var_type maxx = 1.0;
    constexpr var_type maxy = 1.0;
    constexpr var_type maxz = 4.0;

    ///DO NOT MODIFY - 1/max-min pre-calculated
    constexpr var_type NoverDx = var_type(Nx)/(maxx-minx);
    constexpr var_type NoverDy = var_type(Ny)/(maxy-miny);
    constexpr var_type NoverDz = var_type(Nz)/(maxz-minz);
}

#endif