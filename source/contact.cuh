/**
 * @file contact.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Contact search algorithms
 * @date 2023.09.12.
 * 
 * Contains the contact search algorithms. The following contact search 
 * algorithms implemented:
 * - BruteForce: calculates ALL possible contacts
 * - DecomposedDomains: only calculates contact if in a neghbouring cell
 * - DecomposedDomainsFast: EXPERIMENTAL
*/

#ifndef contact_H
#define contact_H

#include "particle.cuh"
#include "material.cuh"
#include "math.cuh"
#include "registers.cuh"
#include "settings.cuh"

 /**
 * \brief Contact data between particles, stored in the registers (preferably)
 */
struct contact
{
    ///tid of the particle we are in contact with
    int tid[MaxContactNumber];

    ///tid of the particle of last contact
    int tid_last[MaxContactNumber];

    ///type of other material
    int material[MaxContactNumber];

    ///equivalent radius of contact
    var_type Rstar[MaxContactNumber];

    ///equiavlent mass of contact
    var_type mstar[MaxContactNumber];

    ///unit vector between particles
    vec3D r[MaxContactNumber];

    ///contact position
    vec3D p[MaxContactNumber];

    ///normal overlap
    var_type deltan[MaxContactNumber];

    ///tangential overlap
    vec3D deltat[MaxContactNumber];

    ///tangential overlap in the last step
    vec3D deltat_last[MaxContactNumber];

    ///number of contacts
    int count;
};


 /**
 * \brief Contact handling of particles
 */
namespace contactHandling
{
    ///calculate neighbours on compile time
    __device__ constexpr int Neighbours[27] = {
        0, 1, -1, DecomposedDomainsConstants::Nx,    -DecomposedDomainsConstants::Nx, 
        DecomposedDomainsConstants::Nx*DecomposedDomainsConstants::Ny,          -DecomposedDomainsConstants::Nx*DecomposedDomainsConstants::Ny,
        DecomposedDomainsConstants::Nx - 1,         -DecomposedDomainsConstants::Nx - 1, 
        DecomposedDomainsConstants::Nx + 1,         -DecomposedDomainsConstants::Nx + 1, 
        DecomposedDomainsConstants::Nx*DecomposedDomainsConstants::Ny + 1,      -DecomposedDomainsConstants::Nx*DecomposedDomainsConstants::Ny + 1,
        DecomposedDomainsConstants::Nx*DecomposedDomainsConstants::Ny - 1,      -DecomposedDomainsConstants::Nx*DecomposedDomainsConstants::Ny - 1,
        DecomposedDomainsConstants::Nx*(1+DecomposedDomainsConstants::Ny),      -DecomposedDomainsConstants::Nx*(1+DecomposedDomainsConstants::Ny),
        DecomposedDomainsConstants::Nx*(1+DecomposedDomainsConstants::Ny)+1,    -DecomposedDomainsConstants::Nx*(1+DecomposedDomainsConstants::Ny)+1,
        DecomposedDomainsConstants::Nx*(1+DecomposedDomainsConstants::Ny)-1,    -DecomposedDomainsConstants::Nx*(1+DecomposedDomainsConstants::Ny)-1,
        DecomposedDomainsConstants::Nx*(-1+DecomposedDomainsConstants::Ny),     -DecomposedDomainsConstants::Nx*(-1+DecomposedDomainsConstants::Ny),
        DecomposedDomainsConstants::Nx*(-1+DecomposedDomainsConstants::Ny)+1,   -DecomposedDomainsConstants::Nx*(-1+DecomposedDomainsConstants::Ny)+1,
        DecomposedDomainsConstants::Nx*(-1+DecomposedDomainsConstants::Ny)-1,   -DecomposedDomainsConstants::Nx*(-1+DecomposedDomainsConstants::Ny)-1
    };

    /**
    * \brief Checks if two cells are neighbours or not
    *
    * @param cid1 Cell id 1
    * @param cid2 Cell id 2
    *
    * @return Returns if the cells are neighbours or not
    */
    __device__ inline bool areNeighbours(int cid1, int cid2)
    {
        if( contactSearch == ContactSearch::DecomposedDomains && DecomposedDomainsConstants::Dimension == 3)
        {
            if(cid1 == cid2 || cid1 == cid2 + Neighbours[1] || cid1 == cid2 + Neighbours[2] ||
                cid1 == cid2 + Neighbours[3] || cid1 == cid2 + Neighbours[4] ||
                cid1 == cid2 + Neighbours[5] || cid1 == cid2 + Neighbours[6] ||
                cid1 == cid2 + Neighbours[7] || cid1 == cid2 + Neighbours[8] ||
                cid1 == cid2 + Neighbours[9] || cid1 == cid2 + Neighbours[10] ||
                cid1 == cid2 + Neighbours[11] || cid1 == cid2 + Neighbours[12] ||
                cid1 == cid2 + Neighbours[13] || cid1 == cid2 + Neighbours[14] ||
                cid1 == cid2 + Neighbours[15] || cid1 == cid2 + Neighbours[16] ||
                cid1 == cid2 + Neighbours[17] || cid1 == cid2 + Neighbours[18] ||
                cid1 == cid2 + Neighbours[19] || cid1 == cid2 + Neighbours[20] ||
                cid1 == cid2 + Neighbours[21] || cid1 == cid2 + Neighbours[22] ||
                cid1 == cid2 + Neighbours[23] || cid1 == cid2 + Neighbours[24] ||
                cid1 == cid2 + Neighbours[25] || cid1 == cid2 + Neighbours[26] )
                return true;
            else return false;
        }

        if( contactSearch == ContactSearch::DecomposedDomains && DecomposedDomainsConstants::Dimension == 2)
        {
            if(cid1 == cid2 || cid1 == cid2 + Neighbours[1] || cid1 == cid2 + Neighbours[2] ||
                cid1 == cid2 + Neighbours[3] || cid1 == cid2 + Neighbours[4] ||
                cid1 == cid2 + Neighbours[7] || cid1 == cid2 + Neighbours[8] ||
                cid1 == cid2 + Neighbours[9] || cid1 == cid2 + Neighbours[10] )
                return true;
            else return false;
        }

        if( contactSearch == ContactSearch::DecomposedDomains && DecomposedDomainsConstants::Dimension == 1)
        {
            if(cid1 == cid2 || cid1 == cid2 + Neighbours[1] || cid1 == cid2 + Neighbours[2])
                return true;
            else return false;
        }

        if( contactSearch == ContactSearch::DecomposedDomainsFast )
        {
            if(cid1 == cid2 || cid1 == cid2 + Neighbours[1] || cid1 == cid2 + Neighbours[2] ||
                cid1 == cid2 + Neighbours[3] || cid1 == cid2 + Neighbours[4]  )
                return true;
            else return false;
        }    
    } 


    /**
    * @brief Calculates the contact parameters between two particles 
    * 
    * @param tid Thread index of the particle
    * @param rmem Register memory containing all the data about the particle
    * @param i Thread index of the particle, particle tid is in contact with
    * @param d Distance between particles
    * @param Rs Sum of radii of particle i and tid
    * @param particles All the particle data
    * @param contacts List of contacts
    * 
    * @return the contact struct is filled up 
    */
    void __device__ CalculateContact(int tid, struct registerMemory &rmem, int i, var_type d, var_type Rs, struct particle particles, struct contact &contacts)
    {
        //data
        contacts.tid[contacts.count] = i;
        contacts.Rstar[contacts.count] = constant::NUMBER_1/(rmem.R_rec + particles.R_rec[i]);
        contacts.mstar[contacts.count] = constant::NUMBER_1/(rmem.m_rec + particles.m_rec[i]);
        contacts.deltan[contacts.count] = Rs - d;

        //material of other
        contacts.material[contacts.count] = particles.material[i];

        //contact position
        contacts.p[contacts.count].x = constant::NUMBER_05*(particles.u.x[i]-rmem.u.x);
        contacts.p[contacts.count].y = constant::NUMBER_05*(particles.u.y[i]-rmem.u.y);
        contacts.p[contacts.count].z = constant::NUMBER_05*(particles.u.z[i]-rmem.u.z);

        //unit vector
        var_type dRec = 1/d;
        contacts.r[contacts.count].x = dRec * (particles.u.x[i]-rmem.u.x);
        contacts.r[contacts.count].y = dRec * (particles.u.y[i]-rmem.u.y);
        contacts.r[contacts.count].z = dRec * (particles.u.z[i]-rmem.u.z);

        //check if they were contact in the last step
        bool wasInContact = false;
        for(int j = 0; j < MaxContactNumber; j++)
        {
            if(i == contacts.tid_last[j]) //they were in contact the last time
            {
                wasInContact = true;
                contacts.deltat[contacts.count].x = contacts.deltat_last[contacts.count].x;
                contacts.deltat[contacts.count].y = contacts.deltat_last[contacts.count].y;
                contacts.deltat[contacts.count].z = contacts.deltat_last[contacts.count].z;
            }
        }

        //if they were not in contact then reset the tangential overlap
        if(wasInContact == false)
        {
            contacts.deltat[contacts.count].x = constant::ZERO;
            contacts.deltat[contacts.count].y = constant::ZERO;
            contacts.deltat[contacts.count].z = constant::ZERO;
        }

        //check end
        contacts.count++; 
        if(contacts.count >= MaxContactNumber)
        {
            contacts.count = MaxContactNumber - 1;
        }
    }//end of calculateContact


    /**
    * @brief Prepares the contact struct for the next timestep by copying deltat into deltat_last for each contact.
    * 
    * @param tid Thread index of the particle
    * @param contacts List of contacts
    */
    void __device__ ResetContacts(int tid, struct contact &contacts)
    {
        //set last tid and deltat_t
        for(int j = 0; j < MaxContactNumber; j++)
        {
            contacts.tid_last[j] = contacts.tid[j];
            contacts.deltat_last[j].x = contacts.deltat[j].x;
            contacts.deltat_last[j].y = contacts.deltat[j].y;
            contacts.deltat_last[j].z = contacts.deltat[j].z;
        }
        contacts.count = 0;
    }

    /**
    * @brief Brute force contact search, which goes through all possible combinations and calculates all contacts
    * 
    * @param tid Thread index of the particle
    * @param rmem Register memory containing all the data about the particle
    * @param numberOfActiveParticles Number of active parameters
    * @param particles All the particle data
    * @param contacts List of contacts
    */
    void __device__ BruteForceContactSearch(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, struct contact &contacts)
    {
        //go through all the particles
        for(int i = 0; i < numberOfActiveParticles; i++)
        {
            var_type d = calculateDistance(rmem.u.x,rmem.u.y,rmem.u.z,particles.u.x[i],particles.u.y[i],particles.u.z[i]);
            var_type Rs = rmem.R + particles.R[i];
            if(d < Rs && tid != i) //contact found
            {
                CalculateContact(tid,rmem,i,d,Rs,particles,contacts);
            }
        }//end of for
    }//end of brute force


    /**
    * @brief Calculates the cell id
    * 
    * @param tid Thread index of the particle
    * @param rmem Register memory containing all the data about the particle
    * @param numberOfActiveParticles Number of active parameters
    * @param particles All the particle data
    * 
    */
    void __device__ CalculateCellId(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles,cooperative_groups::grid_group allThreads)
    {
        //if particle is inactive
        if(tid >= numberOfActiveParticles)
        {
            return;
        }
        int Cx,Cy,Cz;

        //callculate cell coordinates
        Cx = int((rmem.u.x - DecomposedDomainsConstants::minx)*DecomposedDomainsConstants::NoverDx);
        if(Cx < 0) Cx = 0;
        if(Cx >= DecomposedDomainsConstants::Nx) Cx = DecomposedDomainsConstants::Nx-1;
        rmem.cid = Cx;

        if(DecomposedDomainsConstants::Dimension >= 2)
        {
            Cy = int((rmem.u.y - DecomposedDomainsConstants::miny)*DecomposedDomainsConstants::NoverDy);
            if(Cy < 0) Cy = 0;
            if(Cy >= DecomposedDomainsConstants::Ny) Cy = DecomposedDomainsConstants::Ny-1;
            rmem.cid = Cx + DecomposedDomainsConstants::Nx * Cy;
        }

        if(DecomposedDomainsConstants::Dimension >= 3)
        {
            Cz = int((rmem.u.z - DecomposedDomainsConstants::minz)*DecomposedDomainsConstants::NoverDz);
            if(Cz < 0) Cz = 0;
            if(Cz >= DecomposedDomainsConstants::Nz) Cz = DecomposedDomainsConstants::Nz-1;
            rmem.cid = Cx + DecomposedDomainsConstants::Nx * Cy +  DecomposedDomainsConstants::Nx * DecomposedDomainsConstants::Ny * Cz;
        }


        //write to global memory
        particles.cid[tid] = rmem.cid;

        if(UseGPUWideThreadSync)
        {
            allThreads.sync();
        }
        else
        {
            __syncthreads();
        }

    }//end of CalculateCellId

    /**
    * @brief Calculates the cell id for the linked cell algorithm
    * 
    * @param tid Thread index of the particle
    * @param rmem Register memory containing all the data about the particle
    * @param numberOfActiveParticles Number of active parameters
    * @param particles All the particle data
    * 
    */
    void __device__ CalculateCellIdLinkedCells(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles,cooperative_groups::grid_group allThreads)
    {
        //if particle is inactive
        if(tid >= numberOfActiveParticles)
        {
            return;
        }
        int Cx,Cy,Cz;

        //callculate cell coordinates
        Cx = int((rmem.u.x - DecomposedDomainsConstants::minx)*DecomposedDomainsConstants::NoverDx);
        Cy = int((rmem.u.y - DecomposedDomainsConstants::miny)*DecomposedDomainsConstants::NoverDy);
        Cz = int((rmem.u.z - DecomposedDomainsConstants::minz)*DecomposedDomainsConstants::NoverDz);

        //apply limits
        if(Cx < 0) Cx = 0;
        if(Cx >= DecomposedDomainsConstants::Nx) Cx = DecomposedDomainsConstants::Nx-1;
        if(Cy < 0) Cy = 0;
        if(Cy >= DecomposedDomainsConstants::Ny) Cy = DecomposedDomainsConstants::Ny-1;
        rmem.cid = Cx;
        if(Cz < 0) Cz = 0;
        if(Cz >= DecomposedDomainsConstants::Nz) Cz = DecomposedDomainsConstants::Nz-1;

        //calculate cell id
        rmem.cid = Cx + DecomposedDomainsConstants::Nx * Cy +  DecomposedDomainsConstants::Nx * DecomposedDomainsConstants::Ny * Cz;

        //write to global memory
        particles.cid[tid] = rmem.cid;

        //get the id in cell, and increment it
        int idInCell = particles.NinCell[rmem.cid];
        //printf("Cell = %d\t idInCell=%d\n",rmem.cid,idInCell);
        particles.NinCell[rmem.cid] = particles.NinCell[rmem.cid] + 1;
        if(particles.NinCell[rmem.cid] >= DecomposedDomainsConstants::NpCellMax) 
        {
            particles.NinCell[rmem.cid] = DecomposedDomainsConstants::NpCellMax - 1;
        }

        //save the particle in the linked cell list
        particles.linkedCellList[rmem.cid*DecomposedDomainsConstants::NpCellMax + idInCell] = tid;


        if(UseGPUWideThreadSync)
        {
            allThreads.sync();
        }
        else
        {
            __syncthreads();
        }

    }//end of CalculateCellId

    /**
    * @brief Decomposed domains contact search, which checks if particles are in the same or neighbouring cells and calculates all contacts
    * 
    * @param tid Thread index of the particle
    * @param rmem Register memory containing all the data about the particle
    * @param numberOfActiveParticles Number of active parameters
    * @param particles All the particle data
    * @param contacts List of contacts
    */
    void __device__ DecomposedDomainsContactSearch(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, struct contact &contacts)
    {
        //go through all the particles
        for(int i = 0; i < numberOfActiveParticles; i++)
        { 
            int cid = particles.cid[i];
            if( areNeighbours(rmem.cid,cid)) //if other particle is in a neighbouring cell
            {
                var_type d = calculateDistance(rmem.u.x,rmem.u.y,rmem.u.z,particles.u.x[i],particles.u.y[i],particles.u.z[i]);
                var_type Rs = rmem.R + particles.R[i];
                if(d < Rs && tid != i) //contact found
                {
                    CalculateContact(tid,rmem,i,d,Rs,particles,contacts);
                }
            }
        }//end of for
    }//end of brute force


    /**
    * @brief Decomposed domains contact search with linked cell lists, which checks if particles are in the same or neighbouring cells and calculates only these contacts
    * 
    * @param tid Thread index of the particle
    * @param rmem Register memory containing all the data about the particle
    * @param numberOfActiveParticles Number of active parameters
    * @param particles All the particle data
    * @param contacts List of contacts
    */
    void __device__ LinkedCellListContactSearch(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, struct contact &contacts)
    {
        //go through neighbouring cells
        for(int i = 0; i < 27; i++)
        {
            //cell id of neighbour
            int cid = rmem.cid + Neighbours[i];

            if(cid >= 0 && cid < DecomposedDomainsConstants::Ncell)
            {
                for(int j = 0; j < particles.NinCell[cid]; j++)
                {
                    int idx = particles.linkedCellList[cid*DecomposedDomainsConstants::NpCellMax + j];

                    var_type d = calculateDistance(rmem.u.x,rmem.u.y,rmem.u.z,particles.u.x[idx],particles.u.y[idx],particles.u.z[idx]);
                    var_type Rs = rmem.R + particles.R[idx];
                    if(d < Rs && tid != idx) //contact found
                    {
                        CalculateContact(tid,rmem,idx,d,Rs,particles,contacts);
                    }
                }
            }
        }
    }//end of brute force


    /**
    * @brief Initializes the contacts struct at the beginning of the solver kernel
    * 
    * @param tid Thread index of the particle
    * @param contacts List of particles we are in contact with
    *
    * Initalizes tid with -1 and deltat with 0, resets contacts.count to 0
    */
    void __device__ initializeContacts(int tid, struct contact &contacts)
    {
        for(int j = 0; j < MaxContactNumber; j++)
        {
            contacts.tid[j] = -100000;
            contacts.tid_last[j] = -110000;
            contacts.deltat[j].x = constant::ZERO;
            contacts.deltat[j].y = constant::ZERO;
            contacts.deltat[j].z = constant::ZERO;
            contacts.deltat_last[j].x = constant::ZERO;
            contacts.deltat_last[j].y = constant::ZERO;
            contacts.deltat_last[j].z = constant::ZERO;
            contacts.count = 0;
        }
    }

}//end of namespace


#endif