/**
 * @file registers.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Contains the register struct
 * @date 2023.07.24.
 * 
 * This data is stored in the registers!
*/

#ifndef register_H
#define register_H

#include "particle.cuh"

struct registerMemory
{
    //local copy of position
    struct coordinate u;

    //local copy of velocity
    struct coordinate v;

    //local copy of acceleration
    struct coordinate a[AccelerationStored];

    //local copy of angular velocity
    struct coordinate omega;

    //local copy of angular acceleration
    struct coordinate beta[AccelerationStored];

    //local copy of force
    struct coordinate F;

    //local copy of moments
    struct coordinate M;

    //local copy of R
    var_type R;

    //local copy of m
    var_type m;

    //local copy of m_rec
    var_type m_rec;

    //local copy of theta_rec
    var_type theta_rec;

    //local copy of R_rec
    var_type R_rec;

    //material id
    int material;

    //local copy of cid
    int cid;
};

namespace registerHandling{

    __device__ inline void fillRegisterMemory(int tid, struct registerMemory &rmem, struct particle particles)
    {
        //vectors
        rmem.u.x = particles.u.x[tid];
        rmem.u.y = particles.u.y[tid];
        rmem.u.z = particles.u.z[tid];
        rmem.v.x = particles.v.x[tid];
        rmem.v.y = particles.v.y[tid];
        rmem.v.z = particles.v.z[tid];

        rmem.omega.x = particles.omega.x[tid];
        rmem.omega.y = particles.omega.y[tid];
        rmem.omega.z = particles.omega.z[tid];

        //scalars
        rmem.material = particles.material[tid];
        rmem.R = particles.R[tid];
        rmem.R_rec = particles.R_rec[tid];
        rmem.m = particles.m[tid];
        rmem.m_rec = particles.m_rec[tid];
        rmem.theta_rec = particles.theta_rec[tid];

        if(AccelerationStored == 2)
        {
            rmem.a[1].x = particles.a.x[tid + NumberOfParticles]; 
            rmem.a[1].x = particles.a.y[tid + NumberOfParticles];
            rmem.a[1].x = particles.a.z[tid + NumberOfParticles];

            rmem.beta[1].x = particles.beta.x[tid + NumberOfParticles];
            rmem.beta[1].y = particles.beta.y[tid + NumberOfParticles];
            rmem.beta[1].z = particles.beta.z[tid + NumberOfParticles];
        }


    }

    __device__ inline void endOfStepSync(int tid, struct registerMemory &rmem, struct particle particles)
    {
        //position
        particles.u.x[tid] = rmem.u.x;
        particles.u.y[tid] = rmem.u.y;
        particles.u.z[tid] = rmem.u.z;

        //velocity
        particles.v.x[tid] = rmem.v.x;
        particles.v.y[tid] = rmem.v.y;
        particles.v.z[tid] = rmem.v.z;
        particles.omega.x[tid] = rmem.omega.x;
        particles.omega.y[tid] = rmem.omega.y;
        particles.omega.z[tid] = rmem.omega.z;

        //last acceleration
        if(AccelerationStored == 2)
        {
            particles.a.x[tid + NumberOfParticles] = rmem.a[1].x;
            particles.a.y[tid + NumberOfParticles] = rmem.a[1].y;
            particles.a.z[tid + NumberOfParticles] = rmem.a[1].z;

            particles.beta.x[tid + NumberOfParticles] = rmem.beta[1].x;
            particles.beta.y[tid + NumberOfParticles] = rmem.beta[1].y;
            particles.beta.z[tid + NumberOfParticles] = rmem.beta[1].z;
        }
        
    }


    __device__ inline void endOfKernelSync(int tid, struct registerMemory &rmem, struct particle particles)
    {
        if(SaveForce)
        {
            particles.F.x[tid] = rmem.F.x;
            particles.F.y[tid] = rmem.F.y;
            particles.F.z[tid] = rmem.F.z;
        }

        if(SaveTorque)
        {
            particles.M.x[tid] = rmem.M.x;
            particles.M.y[tid] = rmem.M.y;
            particles.M.z[tid] = rmem.M.z;
        }
    }

}//end of namespace

#endif