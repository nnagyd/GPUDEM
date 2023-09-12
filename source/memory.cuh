/**
 * @file memory.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Memory allocation and synchronization of host and device side
 * @date 2023.09.12.
 * 
 * Functions to synchronize data from H2D and D2H
*/

#ifndef memory_H
#define memory_H

#include "math.cuh"

/**
 * \brief Memory handling functions which copy between CPU and GPU and allocate memory
*/
namespace memoryHandling
{
    enum listOfVariables{ All, Position, Velocity, AngularVelocity, Acceleration, AngularAcceleration, Material, Radius, Force, Torque, CellID};

    /**
     * @brief Fills the host side for the particles with zeros
     * 
     * @param particlesH The particle struct which needs to be initalized
     */
     void initializeHostParticles(struct particle &particlesH)
     {
        for(int i = 0; i < NumberOfParticles; i++)
        {
            particlesH.u.x[i] = constant::ZERO;
            particlesH.u.y[i] = constant::ZERO;
            particlesH.u.z[i] = constant::ZERO;
            particlesH.v.x[i] = constant::ZERO;
            particlesH.v.y[i] = constant::ZERO;
            particlesH.v.z[i] = constant::ZERO;
            particlesH.a.x[i] = constant::ZERO;
            particlesH.a.y[i] = constant::ZERO;
            particlesH.a.z[i] = constant::ZERO;
            particlesH.omega.x[i] = constant::ZERO;
            particlesH.omega.y[i] = constant::ZERO;
            particlesH.omega.z[i] = constant::ZERO;
            particlesH.beta.x[i] = constant::ZERO;
            particlesH.beta.y[i] = constant::ZERO;
            particlesH.beta.z[i] = constant::ZERO;
            particlesH.F.x[i] = constant::ZERO;
            particlesH.F.y[i] = constant::ZERO;
            particlesH.F.z[i] = constant::ZERO;
            particlesH.M.x[i] = constant::ZERO;
            particlesH.M.y[i] = constant::ZERO;
            particlesH.M.z[i] = constant::ZERO;
            particlesH.R[i] = constant::ZERO;
            particlesH.R_rec[i] = constant::ZERO;
            particlesH.m[i] = constant::ZERO;
            particlesH.m_rec[i] = constant::ZERO;
            particlesH.theta[i] = constant::ZERO;
            particlesH.theta_rec[i] = constant::ZERO;
            particlesH.material[i] = -1;
            particlesH.cid[i] = 1;
        }

     }


    /**
     * @brief Allocates the host side for the particles
     * 
     * @param particlesH The particle struct which needs memory allocation
     */
    void allocateHostParticles(struct particle &particlesH)
    {
        particlesH.u.x = new var_type[NumberOfParticles];
        particlesH.u.y = new var_type[NumberOfParticles];
        particlesH.u.z = new var_type[NumberOfParticles];
        particlesH.v.x = new var_type[NumberOfParticles];
        particlesH.v.y = new var_type[NumberOfParticles];
        particlesH.v.z = new var_type[NumberOfParticles];
        particlesH.a.x = new var_type[NumberOfParticles*AccelerationStored];
        particlesH.a.y = new var_type[NumberOfParticles*AccelerationStored];
        particlesH.a.z = new var_type[NumberOfParticles*AccelerationStored];
        particlesH.omega.x = new var_type[NumberOfParticles];
        particlesH.omega.y = new var_type[NumberOfParticles];
        particlesH.omega.z = new var_type[NumberOfParticles];
        particlesH.beta.x = new var_type[NumberOfParticles*AccelerationStored];
        particlesH.beta.y = new var_type[NumberOfParticles*AccelerationStored];
        particlesH.beta.z = new var_type[NumberOfParticles*AccelerationStored];
        particlesH.F.x = new var_type[NumberOfParticles];
        particlesH.F.y = new var_type[NumberOfParticles];
        particlesH.F.z = new var_type[NumberOfParticles];
        particlesH.M.x = new var_type[NumberOfParticles];
        particlesH.M.y = new var_type[NumberOfParticles];
        particlesH.M.z = new var_type[NumberOfParticles];
        particlesH.R = new var_type[NumberOfParticles];
        particlesH.R_rec = new var_type[NumberOfParticles];
        particlesH.m = new var_type[NumberOfParticles];
        particlesH.m_rec = new var_type[NumberOfParticles];
        particlesH.theta = new var_type[NumberOfParticles];
        particlesH.theta_rec = new var_type[NumberOfParticles];
        particlesH.material = new int[NumberOfParticles];
        particlesH.cid = new int[NumberOfParticles];


        initializeHostParticles(particlesH);
    }


    /**
     * @brief Free the hist side
     * 
     * @param particlesH The particle struct which is freed
     */
     void freeHostParticles(struct particle &particlesH)
     {
        delete particlesH.u.x;
        delete particlesH.u.y;
        delete particlesH.u.z;
        delete particlesH.v.x;
        delete particlesH.v.y;
        delete particlesH.v.z;
        delete particlesH.a.x;
        delete particlesH.a.y;
        delete particlesH.a.z;
        delete particlesH.omega.x;
        delete particlesH.omega.y;
        delete particlesH.omega.z;
        delete particlesH.beta.x;
        delete particlesH.beta.y;
        delete particlesH.beta.z;
        delete particlesH.F.x;
        delete particlesH.F.y;
        delete particlesH.F.z;
        delete particlesH.M.x;
        delete particlesH.M.y;
        delete particlesH.M.z;
        delete particlesH.R;
        delete particlesH.R_rec;
        delete particlesH.m;
        delete particlesH.m_rec;
        delete particlesH.theta;
        delete particlesH.theta_rec;
        delete particlesH.material;
        delete particlesH.cid;
     }

    /**
     * @brief Allocates the device side for the particles
     * 
     * @param particlesD The particle struct which needs memory allocation
     */
     void allocateDeviceParticles(struct particle &particlesD)
     {
        size_t memorySize = sizeof(var_type) * NumberOfParticles;
        cudaMalloc((void**)&particlesD.u.x, memorySize);
        cudaMalloc((void**)&particlesD.u.y, memorySize);
        cudaMalloc((void**)&particlesD.u.z, memorySize);
        cudaMalloc((void**)&particlesD.v.x, memorySize);
        cudaMalloc((void**)&particlesD.v.y, memorySize);
        cudaMalloc((void**)&particlesD.v.z, memorySize);
        cudaMalloc((void**)&particlesD.a.x, memorySize*AccelerationStored);
        cudaMalloc((void**)&particlesD.a.y, memorySize*AccelerationStored);
        cudaMalloc((void**)&particlesD.a.z, memorySize*AccelerationStored);
        cudaMalloc((void**)&particlesD.omega.x, memorySize);
        cudaMalloc((void**)&particlesD.omega.y, memorySize);
        cudaMalloc((void**)&particlesD.omega.z, memorySize);
        cudaMalloc((void**)&particlesD.beta.x, memorySize*AccelerationStored);
        cudaMalloc((void**)&particlesD.beta.y, memorySize*AccelerationStored);
        cudaMalloc((void**)&particlesD.beta.z, memorySize*AccelerationStored);
        cudaMalloc((void**)&particlesD.F.x, memorySize);
        cudaMalloc((void**)&particlesD.F.y, memorySize);
        cudaMalloc((void**)&particlesD.F.z, memorySize);
        cudaMalloc((void**)&particlesD.M.x, memorySize);
        cudaMalloc((void**)&particlesD.M.y, memorySize);
        cudaMalloc((void**)&particlesD.M.z, memorySize);
        cudaMalloc((void**)&particlesD.R, memorySize); 
        cudaMalloc((void**)&particlesD.R_rec, memorySize); 
        cudaMalloc((void**)&particlesD.m, memorySize);
        cudaMalloc((void**)&particlesD.m_rec, memorySize);
        cudaMalloc((void**)&particlesD.theta, memorySize);
        cudaMalloc((void**)&particlesD.theta_rec, memorySize);
        cudaMalloc((void**)&particlesD.material, sizeof(int) * NumberOfParticles);
        cudaMalloc((void**)&particlesD.cid, sizeof(int) * NumberOfParticles);
     }


    /**
    * @brief Allocates the device side for the particles
    * 
    * @param particlesD The particle struct which needs memory allocation
    */
    void freeDeviceParticles(struct particle &particlesD)
    {
        cudaFree(particlesD.u.x);
        cudaFree(particlesD.u.y);
        cudaFree(particlesD.u.z);
        cudaFree(particlesD.v.x);
        cudaFree(particlesD.v.y);
        cudaFree(particlesD.v.z);
        cudaFree(particlesD.a.x);
        cudaFree(particlesD.a.y);
        cudaFree(particlesD.a.z);
        cudaFree(particlesD.omega.x);
        cudaFree(particlesD.omega.y);
        cudaFree(particlesD.omega.z);
        cudaFree(particlesD.beta.x);
        cudaFree(particlesD.beta.y);
        cudaFree(particlesD.beta.z);
        cudaFree(particlesD.F.x);
        cudaFree(particlesD.F.y);
        cudaFree(particlesD.F.z);
        cudaFree(particlesD.M.x);
        cudaFree(particlesD.M.y);
        cudaFree(particlesD.M.z);
        cudaFree(particlesD.R);
        cudaFree(particlesD.R_rec);
        cudaFree(particlesD.m);
        cudaFree(particlesD.m_rec);
        cudaFree(particlesD.theta);
        cudaFree(particlesD.theta_rec);
        cudaFree(particlesD.material);
        cudaFree(particlesD.cid);
    }


    /**
     * @brief Synchronizes the memory between host and device
     * 
     * @param dest Destination of the data
     * @param source Source of the data
     * @param vars Variables to copy according to the enum
     * @param kind cudaMemcpyDeviceToHost or HostToDevice
     */
    void synchronizeParticles(struct particle dest, struct particle source, listOfVariables vars, cudaMemcpyKind kind)
    {
        int memorySize = NumberOfParticles * sizeof(var_type);
        if(vars == Position || vars == All)
        {
            cudaMemcpy(dest.u.x,source.u.x,memorySize,kind);
            cudaMemcpy(dest.u.y,source.u.y,memorySize,kind);
            cudaMemcpy(dest.u.z,source.u.z,memorySize,kind);
        }

        if(vars == Velocity || vars == All)
        {
            cudaMemcpy(dest.v.x,source.v.x,memorySize,kind);
            cudaMemcpy(dest.v.y,source.v.y,memorySize,kind);
            cudaMemcpy(dest.v.z,source.v.z,memorySize,kind);
        }

        if(vars == AngularVelocity || vars == All)
        {
            cudaMemcpy(dest.omega.x,source.omega.x,memorySize,kind);
            cudaMemcpy(dest.omega.y,source.omega.y,memorySize,kind);
            cudaMemcpy(dest.omega.z,source.omega.z,memorySize,kind);
        }

        if(vars == Acceleration || vars == All)
        {
            cudaMemcpy(dest.a.x,source.a.x,memorySize*AccelerationStored,kind);
            cudaMemcpy(dest.a.y,source.a.y,memorySize*AccelerationStored,kind);
            cudaMemcpy(dest.a.z,source.a.z,memorySize*AccelerationStored,kind);
        }

        if(vars == AngularAcceleration || vars == All)
        {
            cudaMemcpy(dest.beta.x,source.beta.x,memorySize*AccelerationStored,kind);
            cudaMemcpy(dest.beta.y,source.beta.y,memorySize*AccelerationStored,kind);
            cudaMemcpy(dest.beta.z,source.beta.z,memorySize*AccelerationStored,kind);
        }

        if(vars == Radius || vars == All)
        {
            cudaMemcpy(dest.R,source.R,memorySize,kind);
            cudaMemcpy(dest.R_rec,source.R_rec,memorySize,kind);
        }

        if(vars == Material || vars == All)
        {
            cudaMemcpy(dest.m,source.m,memorySize,kind);
            cudaMemcpy(dest.m_rec,source.m_rec,memorySize,kind);
            cudaMemcpy(dest.theta,source.theta,memorySize,kind);
            cudaMemcpy(dest.theta_rec,source.theta_rec,memorySize,kind);
            cudaMemcpy(dest.material,source.material,NumberOfParticles * sizeof(int),kind);
        }

        if(vars == Force || vars == All)
        {
            cudaMemcpy(dest.F.x,source.F.x,memorySize,kind);
            cudaMemcpy(dest.F.y,source.F.y,memorySize,kind);
            cudaMemcpy(dest.F.z,source.F.z,memorySize,kind);
        }

        if(vars == Torque || vars == All)
        {
            cudaMemcpy(dest.M.x,source.M.x,memorySize,kind);
            cudaMemcpy(dest.M.y,source.M.y,memorySize,kind);
            cudaMemcpy(dest.M.z,source.M.z,memorySize,kind);
        }

        if(vars == CellID || vars == All)
        {
            cudaMemcpy(dest.cid,source.cid,NumberOfParticles * sizeof(int),kind);
        }

        CHECK(cudaDeviceSynchronize());
    }
}//end of namespace


#endif

