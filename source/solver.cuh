/**
 * @file solver.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Solver algorithm on the device
 * @date 2023.07.20.
 * 
 * GPU Code for the perThread calculations for each particle
*/

#ifndef solver_H
#define solver_H


#include <cooperative_groups.h>
#include <cuda_runtime_api.h> 
#include <cuda.h> 

#include "particle.cuh"
#include "settings.cuh"
#include "contact.cuh"
#include "domain.cuh"
#include "forces.cuh"
#include "material.cuh"
#include "memory.cuh"
#include "integrate.cuh"
#include "acceleration.cuh"
#include "math.cuh"
#include "io.cuh"
#include "registers.cuh"

/**
    * @brief Kernel using the perThread approach
    * 
    * @param particles All the particle data
    * @param numberOfActiveParticles Number of currently active particles
    * @param pars Material parameters 
    * @param timestep Timestep settings 
    * @param bodyForces Body forces (e.g. gravity)
    * @param boundaryConditions Boundary conditions and types
    * @param launch Number of the kernel launch
    * 
*/
__global__ void solver(struct particle particles, int numberOfActiveParticles, struct materialParameters pars, struct timestepping timestep, struct bodyForce bodyForces, struct boundaryCondition boundaryConditions, int launch)
{
    cooperative_groups::grid_group allThreads = cooperative_groups::this_grid();
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    //possibility to not use all the particles in the simulation
    if(tid >= numberOfActiveParticles)
    {
        return;
    }

    //load registers
    struct registerMemory rmem;
    registerHandling::fillRegisterMemory(tid,rmem,particles);

    //create contacts
    struct contact contacts;
    contactHandling::initializeContacts(tid, contacts);
    
    for(int i = launch*timestep.saveSteps; i < (launch+1)*timestep.saveSteps; i++)
    {
        //0. Start of the step
        contactHandling::ResetContacts(tid,contacts);
        rmem.F.x = constant::ZERO;
        rmem.F.y = constant::ZERO;
        rmem.F.z = constant::ZERO;
        rmem.M.x = constant::ZERO;
        rmem.M.y = constant::ZERO;
        rmem.M.z = constant::ZERO;
        //reset cells
        if(contactSearch == ContactSearch::LinkedCellList)
        {
            int idx = tid;
            while(idx < DecomposedDomainsConstants::Ncell * DecomposedDomainsConstants::NpCellMax)
            {
                if(idx < DecomposedDomainsConstants::Ncell)
                {
                    particles.NinCell[idx] = 0;
                }
                particles.linkedCellList[idx] = 0;
                idx += numberOfActiveParticles;
            }
        }

        //1. boundary conditions 
        if(NumberOfBoundaries > 0)
        {
            domainHandling::applyBoundaryConditions(tid,rmem,particles,boundaryConditions,contacts,pars,timestep);
        }

        //2. contact search
        if(contactSearch == ContactSearch::BruteForce)
        {
            contactHandling::BruteForceContactSearch(tid,rmem,numberOfActiveParticles,particles,contacts);
        }
        if(contactSearch == ContactSearch::DecomposedDomains || contactSearch == ContactSearch::DecomposedDomainsFast)
        {
            contactHandling::CalculateCellId(tid,rmem,numberOfActiveParticles,particles,allThreads);
            contactHandling::DecomposedDomainsContactSearch(tid,rmem,numberOfActiveParticles,particles,contacts);
        }
        if(contactSearch == ContactSearch::LinkedCellList)
        {
            contactHandling::CalculateCellIdLinkedCells(tid,rmem,numberOfActiveParticles,particles,allThreads);
            contactHandling::LinkedCellListContactSearch(tid,rmem,numberOfActiveParticles,particles,contacts);
        }

        //3. calculate forces
        if(contactModel == ContactModel::Mindlin)
        {
            forceHandling::calculateForceMindlin(tid,rmem,particles,contacts, pars, timestep);
        }

        //4. calculate acceleration
        accelerationHandling::calculateDefault(tid, rmem, particles);
        if(BodyForce) //add the acceleration as a result of gravity
        {
            accelerationHandling::addBodyForces(tid, rmem, particles, bodyForces);
        }
        if(UseGPUWideThreadSync)
        {
            allThreads.sync();
        }
        else
        {
            __syncthreads();
        }
    

        //5. timestepping 
        if(timeIntegration == TimeIntegration::Euler)
        {
            integrators::euler(tid,rmem,particles,timestep);
        }
        if(timeIntegration == TimeIntegration::Exact)
        {
            integrators::exact(tid,rmem,particles,timestep);
        }
        if(timeIntegration == TimeIntegration::Adams2)
        {
            integrators::adams2(tid,rmem,particles,timestep,i);
        }

        //6. Synchronize registers and global memory
        registerHandling::endOfStepSync(tid,rmem,particles);
        if(UseGPUWideThreadSync)
        {
            allThreads.sync();
        }
        else
        {
            __syncthreads();
        }
    }//end of timesteps

    //synchronize register data to global memory if data is saved
    registerHandling::endOfKernelSync(tid,rmem,particles);


    
}//end of kernel

#endif