/**
 * @file contact.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Contact search algorithms
 * @date 2023.07.24.
 * 
 * Contains the contact search algorithms
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

    ///equivalent radius of contact
    var_type Rstar[MaxContactNumber];

    ///equiavlent mass of contact
    var_type mstar[MaxContactNumber];

    ///equiavlent E of contact
    var_type Estar[MaxContactNumber];

    ///equiavlent E of contact
    var_type mustar[MaxContactNumber];

    ///equiavlent E of contact
    var_type mu0star[MaxContactNumber];

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
    //calculate neighbours on compile time
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

    __device__ inline bool areNeighbours(int cid1, int cid2)
    {
        if( contactSearch == ContactSearch::DecomposedDomains )
        {
            if(cid1 == cid2 || cid1 == cid2 + Neighbours[1] || cid1 == cid2 + Neighbours[2] ||
                cid2 + Neighbours[3] || cid1 == cid2 + Neighbours[4] ||
                cid2 + Neighbours[5] || cid1 == cid2 + Neighbours[6] ||
                cid2 + Neighbours[7] || cid1 == cid2 + Neighbours[8] ||
                cid2 + Neighbours[9] || cid1 == cid2 + Neighbours[10] ||
                cid2 + Neighbours[11] || cid1 == cid2 + Neighbours[12] ||
                cid2 + Neighbours[13] || cid1 == cid2 + Neighbours[14] ||
                cid2 + Neighbours[15] || cid1 == cid2 + Neighbours[16] ||
                cid2 + Neighbours[17] || cid1 == cid2 + Neighbours[18] ||
                cid2 + Neighbours[19] || cid1 == cid2 + Neighbours[20] ||
                cid2 + Neighbours[21] || cid1 == cid2 + Neighbours[22] ||
                cid2 + Neighbours[23] || cid1 == cid2 + Neighbours[24] ||
                cid2 + Neighbours[25] || cid1 == cid2 + Neighbours[26] )
                return true;
            else return false;
        }
        if( contactSearch == ContactSearch::DecomposedDomainsFast )
        {
            if(cid1 == cid2 || cid1 == cid2 + Neighbours[1] || cid1 == cid2 + Neighbours[2] ||
                cid2 + Neighbours[3] || cid1 == cid2 + Neighbours[4]  )
                return true;
            else return false;
        }    
    } 


        /**
    * @brief Brute force contact search, which goes through all possible combinations 
    * 
    * @param tid Thread index of the particle
    * @param x,y,z Coordinates of particle tid
    * @param i Thread index of the particle, particle tid is in contact with
    * @param d Distance between particles
    * @param particles All the particle data
    * @param contacts List of particles we are in contact with
    * 
    */
    void __device__ CalculateContact(int tid, struct registerMemory &rmem, int i, var_type d, var_type Rs, struct particle particles, struct contact &contacts)
    {
        //data
        contacts.tid[contacts.count] = i;
        contacts.Rstar[contacts.count] = constant::NUMBER_1/(rmem.R_rec + particles.R_rec[i]);
        contacts.mstar[contacts.count] = constant::NUMBER_1/(rmem.m_rec + particles.m_rec[i]);

        contacts.Estar[contacts.count] = constant::NUMBER_1/((constant::NUMBER_1 - rmem.nu*rmem.nu)*rmem.E_rec + (constant::NUMBER_1 - particles.nu[i]*particles.nu[i])*particles.E_rec[i]);
        contacts.mustar[contacts.count] = constant::NUMBER_05*(rmem.mu + particles.mu[i]);
        contacts.mu0star[contacts.count] = constant::NUMBER_05*(rmem.mu0 + particles.mu0[i]);
        contacts.deltan[contacts.count] = Rs - d;

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
    }//end of calculateContact


    /**
    * @brief Brute force contact search, which goes through all possible combinations 
    * 
    * @param tid Thread index of the particle
    * @param particles All the particle data
    * @param contacts List of particles we are in contact with
    * 
    */
    void __device__ BruteForceContactSearch(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, struct contact &contacts)
    {
        //set last tid and deltat_t
        for(int j = 0; j < MaxContactNumber; j++)
        {
            contacts.tid_last[j] = contacts.tid[j];
            contacts.deltat_last[j].x = contacts.deltat[j].x;
            contacts.deltat_last[j].y = contacts.deltat[j].y;
            contacts.deltat_last[j].z = contacts.deltat[j].z;
        }

        //go through all the particles
        contacts.count = 0;
        for(int i = 0; i < numberOfActiveParticles; i++)
        {
            var_type d = calculateDistance(rmem.u.x,rmem.u.y,rmem.u.z,particles.u.x[i],particles.u.y[i],particles.u.z[i]);
            var_type Rs = rmem.R + particles.R[i];
            if(d < Rs && tid != i) //contact found
            {
                CalculateContact(tid,rmem,i,d,Rs,particles,contacts);


                if(contacts.count == MaxContactNumber)
                {
                    if(Debug) 
                    {
                        printf("Max contact number reached at thread%d\n",tid);
                    }
                    return;
                }
            }
        }//end of for
    }//end of brute force


    /**
    * @brief Calculates the cell id
    * 
    * @param tid Thread index of the particle
    * @param particles All the particle data
    * 
    */
    void __device__ CalculateCellId(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles)
    {
        //if particle is inactive
        if(tid >= numberOfActiveParticles)
        {
            return;
        }

        //callculate cell coordinates
        int Cx = int((rmem.u.x - DecomposedDomainsConstants::minx)*DecomposedDomainsConstants::NoverDx);
        int Cy = int((rmem.u.y - DecomposedDomainsConstants::miny)*DecomposedDomainsConstants::NoverDy);
        int Cz = int((rmem.u.z - DecomposedDomainsConstants::minz)*DecomposedDomainsConstants::NoverDz);

        //apply cutoffs
        if(Cx < 0) Cx = 0;
        if(Cy < 0) Cy = 0;
        if(Cz < 0) Cz = 0;
        if(Cx >= DecomposedDomainsConstants::Nx) Cx = DecomposedDomainsConstants::Nx-1;
        if(Cy >= DecomposedDomainsConstants::Ny) Cy = DecomposedDomainsConstants::Ny-1;
        if(Cz >= DecomposedDomainsConstants::Nz) Cz = DecomposedDomainsConstants::Nz-1;

        rmem.cid = Cx + DecomposedDomainsConstants::Nx * Cy +  DecomposedDomainsConstants::Nx * DecomposedDomainsConstants::Ny * Cz;

        if(Debug && tid==0)
        {
            printf("u=(%6.2lf,%6.2lf,%6.2lf)\t C=(%d,%d,%d)\t cid=%d\n",rmem.u.x,rmem.u.y,rmem.u.z,Cx,Cy,Cz,rmem.cid);
        }

        //write to global memory
        particles.cid[tid] = rmem.cid;

    }//end of CalculateCellId


    void __device__ DecomposedDomainsContactSearch(int tid, struct registerMemory &rmem, int numberOfActiveParticles, struct particle particles, struct contact &contacts)
    {
        //set last tid and deltat_t
        for(int j = 0; j < MaxContactNumber; j++)
        {
            contacts.tid_last[j] = contacts.tid[j];
            contacts.deltat_last[j].x = contacts.deltat[j].x;
            contacts.deltat_last[j].y = contacts.deltat[j].y;
            contacts.deltat_last[j].z = contacts.deltat[j].z;
        }

        //go through all the particles
        contacts.count = 0;
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

                    if(Debug)
                    {
                        printf("Particles are in contact: %d \t %d\n",rmem.cid,cid);
                    }

                    if(contacts.count == MaxContactNumber)
                    {
                        if(Debug) 
                        {
                            printf("Max contact number reached at thread%d\n",tid);
                        }
                        return;
                    }
                }
            }
        }//end of for
    }//end of brute force


    /**
    * @brief Initializes the contacts struct
    * 
    * @param tid Thread index of the particle
    * @param contacts List of particles we are in contact with
    * 
    */
    void __device__ initializeContacts(int tid, struct contact &contacts)
    {
        for(int j = 0; j < MaxContactNumber; j++)
        {
            contacts.tid[j] = -1;
            contacts.deltat[j].x = constant::ZERO;
            contacts.deltat[j].y = constant::ZERO;
            contacts.deltat[j].z = constant::ZERO;
            contacts.count = 0;
        }
    }

}//end of namespace


#endif