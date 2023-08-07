/**
 * @file acceleration.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Calculates the acceleration
 * @date 2023.07.30.
 * 
 * Functions to do it
*/

#ifndef acceleration_H
#define acceleration_H

#include "particle.cuh"
#include "timestep.cuh"
#include "math.cuh"

namespace accelerationHandling
{
    /**
    * @brief Calculates the acceleration of the particles
    * 
    * @param tid Thread index
    * @param particles Contains all the particle data 
    * 
    * @return Particle accelerations are written in the particles struct
    */
    __device__ void calculateDefault(int tid, struct registerMemory &rmem, struct particle particles)
    {
        //acc. and angular acc.
        rmem.a[0].x =rmem.F.x * rmem.m_rec;
        rmem.a[0].y =rmem.F.y * rmem.m_rec;
        rmem.a[0].z =rmem.F.z * rmem.m_rec;
        rmem.beta[0].x =rmem.M.x * rmem.theta_rec;
        rmem.beta[0].y =rmem.M.y * rmem.theta_rec;
        rmem.beta[0].z =rmem.M.z * rmem.theta_rec;
    }

    /**
    * @brief Modifies the acceleration with the body forces terms
    * 
    * @param tid Thread index
    * @param particles Contains all the particle data 
    * @param bodyForces Contains all the information about the body forces
    * 
    * @return Particle accelerations are written in the particles struct
    */
    __device__ void addBodyForces(int tid, struct registerMemory &rmem, struct particle particles, struct bodyForce bodyForces)
    {
        rmem.a[0].x = rmem.a[0].x + bodyForces.x;
        rmem.a[0].y = rmem.a[0].y + bodyForces.y;
        rmem.a[0].z = rmem.a[0].z + bodyForces.z;
    }
}

#endif