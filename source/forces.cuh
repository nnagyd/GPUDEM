/**
 * @file forces.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Force calculations
 * @date 2023.07.24.
 * 
 * Different methods for force calculations
*/

#ifndef forces_H
#define forces_H

#include "particle.cuh"
#include "math.cuh"
#include "contact.cuh"
#include "timestep.cuh"

/**
 * \brief Stores the constant body forces in the different directions
 */
struct bodyForce
{
    ///x direction
    var_type x;
    ///y direction
    var_type y;
    ///z direction
    var_type z;
};

namespace forceHandling
{
    /**
    * @brief Calculates the force acting on the particle in x,y,z system using the Mindlin-Hertz theory
    * 
    * @param tid Thread index
    * @param particles The particles struct containing all the data about them
    * @param contacts The struct containing all the contacts
    * @param pars The struct containing all the contacts
    * 
    * @return Returns the force in x,y,z coordinate system
    */
    __device__ inline void calculateForceMindlin(int tid, struct registerMemory &rmem, struct particle particles, struct contact contacts, struct materialParameters pars, struct timestepping timestep)
    {
        //read particle data into the registers
        vec3D vs(rmem.v.x,rmem.v.y,rmem.v.z);
        vec3D omegas(rmem.omega.x,rmem.omega.y,rmem.omega.z);

        vec3D Ftotal(constant::ZERO,constant::ZERO,constant::ZERO);
        vec3D Mtotal(constant::ZERO,constant::ZERO,constant::ZERO);


        //go through all the contacts
        for(int i = 0; i < contacts.count; i++)
        {
            vec3D test(constant::ZERO,constant::ZERO,constant::ZERO);
            //particle index of the i.th contact
            int tidi = contacts.tid[i];

            //read the other particles data into the registers
            vec3D vi(particles.v.x[tidi],particles.v.y[tidi],particles.v.z[tidi]);
            vec3D omegai(particles.omega.x[tidi],particles.omega.y[tidi],particles.omega.z[tidi]);
            var_type Ri = particles.R[tidi];

            //kinematics
            vec3D vrel;
            vrel = vs - vi + ((omegas*rmem.R + omegai*Ri) ^ contacts.r[i]);
            var_type vreln_scalar = vrel * contacts.r[i];
            vec3D vreln =  contacts.r[i]*vreln_scalar;
            vec3D vrelt = vrel - vreln;

            //calculate tangential overlap
            contacts.deltat[i] = contacts.deltat[i] + (vrelt * timestep.dt);

            //equivalent shear stiffness
            var_type Gstar = pars.G; //UPDATE

            //equivalent stiffness, normal and tangential
            var_type Rdelta = sqrt(contacts.Rstar[i]*contacts.deltan[i]);
            var_type Sn = constant::NUMBER_2 * contacts.Estar[i] * Rdelta;
            var_type St = constant::NUMBER_8 * Gstar * Rdelta;

            /// FORCES
            //normal elastic force
            var_type Fne_scalar = constant::NUMBER_4o3 * contacts.Estar[i] * Rdelta * contacts.deltan[i];
            vec3D Fne = contacts.r[i]* (-Fne_scalar);

            //normal damping force
            var_type Fnd_scalar = constant::DAMPING * pars.beta * sqrt(Sn * contacts.mstar[i]);
            vec3D Fnd = vreln * Fnd_scalar;

            //tangential elastic force
            vec3D Fte;
            Fte = contacts.deltat[i] * (-St);

            //tangential damping force
            var_type Ftd_scalar = constant::DAMPING * pars.beta * sqrt(St * contacts.mstar[i]);
            vec3D Ftd = vrelt * Ftd_scalar;

            //total normal and tangentional force
            vec3D Fn = Fne + Fnd;
            vec3D Ft = Fte + Ftd;

            //check for sliding
            if(Ft.length() > Fn.length() * contacts.mu0star[i])
            {
                //if sliding
                Ft = Fn * contacts.mustar[i];
            }

            //torque
            vec3D M = contacts.p[i] ^ Ft;

            //force
            vec3D F = Fn + Ft;
            if(Debug == 2)
            {
                printf(
                "tid=%d \t Fn=(%7.3lf,%7.3lf,%7.3lf), Ft=(%7.3lf,%7.3lf,%7.3lf),  F=(%7.3lf,%7.3lf,%7.3lf)\n",
                tid,
                Fn.x,Fn.y,Fn.z,
                Ft.x,Ft.y,Ft.z,
                F.x,F.y,F.z);
            }

            //add the force and torque to the total
            Ftotal = Ftotal + F;
            Mtotal = Mtotal + M;

        }

        //copy force and torque to the particles data
        rmem.F.x = Ftotal.x;
        rmem.F.y = Ftotal.y;
        rmem.F.z = Ftotal.z;
        rmem.M.x = Mtotal.x;
        rmem.M.y = Mtotal.y;
        rmem.M.z = Mtotal.z;
    }



    /**
    * \brief Calculates the total kinetic energy
    *
    * @param particles A list of particles
    */
    var_type calculateTotalKineticEnergy(struct particle particles, int numberOfActiveParticles)
    {
        var_type k = constant::ZERO;
        for(int i = 0; i < numberOfActiveParticles; i++)
        {
            k += constant::NUMBER_05*particles.m[i]*(particles.v.x[i]*particles.v.x[i] + particles.v.y[i]*particles.v.y[i] + particles.v.z[i]*particles.v.z[i]);
        }
        return k;
    }

    /**
     * \brief Calculates the total potential energy
     *
    * @param particles A list of particles
    * @param bodyForces Volumetric forces acting on the particles
    */
    var_type calculateTotalPotentialEnergy(struct particle particles, struct bodyForce bodyForces, int numberOfActiveParticles)
    {
        var_type p = constant::ZERO;
        for(int i = 0; i < numberOfActiveParticles; i++)
        {
            p += -particles.m[i]*(particles.u.x[i]*bodyForces.x + particles.u.y[i]*bodyForces.y + particles.u.z[i]*bodyForces.z);
        }
        return p;
    }
}

#endif