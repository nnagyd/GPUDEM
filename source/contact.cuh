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
#include "mesh_parameters.cuh"

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
    __device__ inline void decodeCellId(int cid, const RuntimeMeshParameters &mesh, int &cx, int &cy, int &cz)
    {
        const int cellsPerLayer = mesh.nx * mesh.ny;
        cz = cid / cellsPerLayer;
        const int inLayer = cid - cz * cellsPerLayer;
        cy = inLayer / mesh.nx;
        cx = inLayer - cy * mesh.nx;
    }

    /**
    * \brief Checks if two cells are neighbours or not
    *
    * @param cid1 Cell id 1
    * @param cid2 Cell id 2
    *
    * @return Returns if the cells are neighbours or not
    */
    __device__ inline bool areNeighbours(int cid1, int cid2, const RuntimeMeshParameters &mesh)
    {
        int cx1, cy1, cz1;
        int cx2, cy2, cz2;
        decodeCellId(cid1, mesh, cx1, cy1, cz1);
        decodeCellId(cid2, mesh, cx2, cy2, cz2);

        int dx = cx1 - cx2;
        int dy = cy1 - cy2;
        int dz = cz1 - cz2;
        if(dx < 0) dx = -dx;
        if(dy < 0) dy = -dy;
        if(dz < 0) dz = -dz;

        if(contactSearch == ContactSearch::DecomposedDomainsFast)
        {
            return (dx <= 1 && dy <= 1 && dz == 0);
        }
        if(DecomposedDomainsConstants::Dimension == 1)
        {
            return (dx <= 1 && dy == 0 && dz == 0);
        }
        if(DecomposedDomainsConstants::Dimension == 2)
        {
            return (dx <= 1 && dy <= 1 && dz == 0);
        }
        return (dx <= 1 && dy <= 1 && dz <= 1);
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
        //check if they were contact in the last step
        bool wasInContact = false;

        contacts.deltan[contacts.count] = Rs - d;
        //printf("deltan = %lf\n",contacts.deltan[contacts.count]);

        //contact position
        contacts.p[contacts.count].x = constant::NUMBER_05*(particles.u.x[i]-rmem.u.x);
        contacts.p[contacts.count].y = constant::NUMBER_05*(particles.u.y[i]-rmem.u.y);
        contacts.p[contacts.count].z = constant::NUMBER_05*(particles.u.z[i]-rmem.u.z);

        //unit vector
        var_type dRec = 1/d;
        contacts.r[contacts.count].x = dRec * (particles.u.x[i]-rmem.u.x);
        contacts.r[contacts.count].y = dRec * (particles.u.y[i]-rmem.u.y);
        contacts.r[contacts.count].z = dRec * (particles.u.z[i]-rmem.u.z);

        //if it was the same contact last time
        if(i == contacts.tid_last[contacts.count]) //they were in contact the last time, exact same data
        {
            wasInContact = true;
        }
        else
        {
            //calculate new parameters
            contacts.tid[contacts.count] = i;
            contacts.Rstar[contacts.count] = constant::NUMBER_1/(rmem.R_rec + particles.R_rec[i]);
            contacts.mstar[contacts.count] = constant::NUMBER_1/(rmem.m_rec + particles.m_rec[i]);

            //material of other
            contacts.material[contacts.count] = particles.material[i];

            for(int j = 0; j < MaxContactNumber; j++)
            {
                if(i == contacts.tid_last[j]) //they were in contact the last time but with different index
                {
                    wasInContact = true;
                    break;
                }
            }
        }

        
        //if they were not in contact then reset the tangential overlap and calculate new parameters otherwise refresh
        if(wasInContact == false)
        {
            contacts.deltat[contacts.count].x = constant::ZERO;
            contacts.deltat[contacts.count].y = constant::ZERO;
            contacts.deltat[contacts.count].z = constant::ZERO;
        }
        else
        {
            contacts.deltat[contacts.count].x = contacts.deltat_last[contacts.count].x;
            contacts.deltat[contacts.count].y = contacts.deltat_last[contacts.count].y;
            contacts.deltat[contacts.count].z = contacts.deltat_last[contacts.count].z;
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
    void __device__ CalculateCellId(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, const RuntimeMeshParameters &mesh)
    {
        //if particle is inactive
        if(tid >= numberOfActiveParticles)
        {
            return;
        }
        int Cx,Cy,Cz;

        //callculate cell coordinates
        Cx = int((rmem.u.x - mesh.minx)*mesh.NoverDx);
        if(Cx < 0) Cx = 0;
        if(Cx >= mesh.nx) Cx = mesh.nx-1;
        rmem.cid = Cx;

        if(DecomposedDomainsConstants::Dimension >= 2)
        {
            Cy = int((rmem.u.y - mesh.miny)*mesh.NoverDy);
            if(Cy < 0) Cy = 0;
            if(Cy >= mesh.ny) Cy = mesh.ny-1;
            rmem.cid = Cx + mesh.nx * Cy;
        }

        if(DecomposedDomainsConstants::Dimension >= 3)
        {
            Cz = int((rmem.u.z - mesh.minz)*mesh.NoverDz);
            if(Cz < 0) Cz = 0;
            if(Cz >= mesh.nz) Cz = mesh.nz-1;
            rmem.cid = Cx + mesh.nx * Cy +  mesh.nx * mesh.ny * Cz;
        }


        //write to global memory
        particles.cid[tid] = rmem.cid;
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
    void __device__ CalculateCellIdLinkedCells(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, const RuntimeMeshParameters &mesh)
    {
        //if particle is inactive
        if(tid >= numberOfActiveParticles)
        {
            return;
        }
        int Cx,Cy,Cz;

        //callculate cell coordinates
        Cx = int((rmem.u.x - mesh.minx)*mesh.NoverDx);
        Cy = int((rmem.u.y - mesh.miny)*mesh.NoverDy);
        Cz = int((rmem.u.z - mesh.minz)*mesh.NoverDz);

        //apply limits
        if(Cx < 0) Cx = 0;
        if(Cx >= mesh.nx) Cx = mesh.nx-1;
        if(Cy < 0) Cy = 0;
        if(Cy >= mesh.ny) Cy = mesh.ny-1;
        rmem.cid = Cx;
        if(Cz < 0) Cz = 0;
        if(Cz >= mesh.nz) Cz = mesh.nz-1;

        //calculate cell id
        rmem.cid = Cx + mesh.nx * Cy +  mesh.nx * mesh.ny * Cz;

        //write to global memory
        particles.cid[tid] = rmem.cid;

        //get the id in cell, and increment it
        int idInCell = atomicInc(&particles.NinCell[rmem.cid],DecomposedDomainsConstants::NpCellMax);

        //printf("Cell = %d\t idInCell=%d\n",rmem.cid,idInCell);
        /*particles.NinCell[rmem.cid] = particles.NinCell[rmem.cid] + 1;
        if(particles.NinCell[rmem.cid] >= DecomposedDomainsConstants::NpCellMax) 
        {
            particles.NinCell[rmem.cid] = DecomposedDomainsConstants::NpCellMax - 1;
        }*/

        //save the particle in the linked cell list
        particles.linkedCellList[rmem.cid*DecomposedDomainsConstants::NpCellMax + idInCell] = tid;
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
    void __device__ DecomposedDomainsContactSearch(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, struct contact &contacts, const RuntimeMeshParameters &mesh)
    {
        //go through all the particles
        for(int i = 0; i < numberOfActiveParticles; i++)
        { 
            int cid = particles.cid[i];
            if( areNeighbours(rmem.cid,cid,mesh)) //if other particle is in a neighbouring cell
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
    void __device__ LinkedCellListContactSearch(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, struct contact &contacts, const RuntimeMeshParameters &mesh)
    {
        int cx, cy, cz;
        decodeCellId(rmem.cid, mesh, cx, cy, cz);

        //go through neighbouring cells
        for(int dz = -1; dz <= 1; dz++)
        {
            if(DecomposedDomainsConstants::Dimension < 3 && dz != 0)
            {
                continue;
            }
            const int ncz = cz + dz;
            if(ncz < 0 || ncz >= mesh.nz)
            {
                continue;
            }

            for(int dy = -1; dy <= 1; dy++)
            {
                if(DecomposedDomainsConstants::Dimension < 2 && dy != 0)
                {
                    continue;
                }
                const int ncy = cy + dy;
                if(ncy < 0 || ncy >= mesh.ny)
                {
                    continue;
                }

                for(int dx = -1; dx <= 1; dx++)
                {
                    const int ncx = cx + dx;
                    if(ncx < 0 || ncx >= mesh.nx)
                    {
                        continue;
                    }

                    const int cid = ncx + mesh.nx * ncy + mesh.nx * mesh.ny * ncz;

                    if(cid >= 0 && cid < mesh.ncell)
                    {
                        for(int j = 0; j < particles.NinCell[cid]; j++)
                        {
                            int idx = particles.linkedCellList[cid*DecomposedDomainsConstants::NpCellMax + j];

                    var_type d_square = calculateDistanceSquare(rmem.u.x,rmem.u.y,rmem.u.z,particles.u.x[idx],particles.u.y[idx],particles.u.z[idx]);
                    var_type Rs = rmem.R + particles.R[idx];
                    var_type dist_check = Rs;

                    if(WaterBridges)
                    {
                        dist_check += WaterBridgeDistanceRange;
                    }


                    //printf("cid = %d tid = %d x=(%lf,%lf,%lf) idx = %d d2 = %lf dist = %lf\n",cid,tid,rmem.u.x,rmem.u.y,rmem.u.z,idx,d_square,dist_check);
                            if(d_square < dist_check*dist_check && tid != idx) //contact found
                            {
                                CalculateContact(tid,rmem,idx,sqrt(d_square),Rs,particles,contacts);
                            }
                        }
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