/**
 * @file forces.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Force calculations
 * @date 2023.09.12.
 * 
 * Contains methods for force and energy calculations
*/

#ifndef forces_H
#define forces_H

#include "particle.cuh"
#include "math.cuh"
#include "contact.cuh"
#include "timestep.cuh"

/**
 * \brief Stores the constant body forces (e.g. gravity) in the different directions
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

/**
 * \brief Contains all the functions to calculate the force between particles
 */
namespace forceHandling
{
    /**
    * @brief Calculates the force acting on the particle in x,y,z system using the Mindlin-Hertz theory
    * 
    * @param tid Thread index
    * @param rmem Register memory containing all the data about the particle
    * @param particles The particles struct containing all the data about them
    * @param contacts The struct containing all the contacts
    * @param pars The struct containing all the material parameters
    * @param timestep Timestep settings
    * 
    * @return Returns the force in x,y,z coordinate system (adds it to rmem)
    */
    __device__ inline void calculateForceMindlin(int tid, struct registerMemory &rmem, struct particle particles, struct contact contacts, struct materialParameters pars, struct timestepping timestep)
    {
        //read particle data into the registers
        vec3D vs(rmem.v.x,rmem.v.y,rmem.v.z);
        vec3D omegas(rmem.omega.x,rmem.omega.y,rmem.omega.z);

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
            vec3D v_rel = vs - vi + ((omegas*rmem.R + omegai*Ri) ^ contacts.r[i]);
            var_type vn_rel_norm = v_rel * contacts.r[i];
            vec3D vn_rel =  contacts.r[i]*vn_rel_norm;
            vec3D vt_rel = v_rel - vn_rel;

            //calculate tangential overlap
            contacts.deltat[i] = contacts.deltat[i] + (vt_rel * timestep.dt);

            //equivalent stiffness, normal and tangential
            var_type Rdelta = sqrt(contacts.Rstar[i]*contacts.deltan[i]);
            var_type Sn = constant::NUMBER_2 * pars.pairing[rmem.material].E_star[contacts.material[i]] * Rdelta;
            var_type St = constant::NUMBER_8 * pars.pairing[rmem.material].G_star[contacts.material[i]] * Rdelta;

            /// FORCES
            //normal elastic force
            var_type Fne_norm = constant::NUMBER_4o3 * pars.pairing[rmem.material].E_star[contacts.material[i]] * Rdelta * contacts.deltan[i];
            vec3D Fne = contacts.r[i]* (-Fne_norm);

            //normal damping force
            var_type Fnd_norm = constant::DAMPING * pars.pairing[rmem.material].beta_star[contacts.material[i]] * sqrt(Sn * contacts.mstar[i]);
            vec3D Fnd = vn_rel * Fnd_norm;

            //tangential elastic force
            vec3D Fte = contacts.deltat[i] * (-St);

            //tangential damping force
            var_type Ftd_norm = constant::DAMPING * pars.pairing[rmem.material].beta_star[contacts.material[i]] * sqrt(St * contacts.mstar[i]);
            vec3D Ftd = vt_rel * Ftd_norm;

            //total normal and tangentional force
            vec3D Fn = Fne + Fnd;
            vec3D Ft = Fte + Ftd;

            //check for sliding
            var_type Ft_norm = Ft.length();
            var_type Fn_norm = Fn.length();
            if(Ft_norm > Fn_norm * pars.pairing[rmem.material].mu0_star[contacts.material[i]])
            {
                //if sliding
                Ft = Ft*(Fn_norm/Ft_norm * pars.pairing[rmem.material].mu_star[contacts.material[i]]);
            }

            //torque
            vec3D M = (contacts.p[i] ^ Ft)*(constant::NUMBER_1);

            //calculate rolling
            if(RollingFriction)
            {
                var_type omegas_norm = omegas.length();
                if(omegas_norm != constant::ZERO)
                {
                    vec3D omegas_unit = omegas * (constant::NUMBER_1 / omegas_norm);
                    vec3D Mr = omegas_unit * (-pars.pairing[rmem.material].mur_star[contacts.material[i]] * Fn_norm * contacts.p[i].length());
                    M = M + Mr;
                }
            }

            //force
            vec3D F = Fn + Ft;

            //add the force and torque to the total
            rmem.F.x += F.x;
            rmem.F.y += F.y;
            rmem.F.z += F.z;
            rmem.M.x += M.x;
            rmem.M.y += M.y;
            rmem.M.z += M.z;
        }
    }



    /**
    * \brief Calculates the total kinetic energy
    *
    * @param particles A list of particles
    * @param numberOfActiveParticles Number of active parameters
    */
    var_type calculateTotalKineticEnergy(struct particle particles, int numberOfActiveParticles)
    {
        var_type k = constant::ZERO;
        for(int i = 0; i < numberOfActiveParticles; i++)
        {
            k += constant::NUMBER_05*particles.m[i]*(particles.v.x[i]*particles.v.x[i] + particles.v.y[i]*particles.v.y[i] + particles.v.z[i]*particles.v.z[i]) + constant::NUMBER_05*particles.theta[i]*(particles.omega.x[i]*particles.omega.x[i] + particles.omega.y[i]*particles.omega.y[i] + particles.omega.z[i]*particles.omega.z[i]);
        }
        return k;
    }

    /**
     * \brief Calculates the total potential energy
     *
    * @param particles A list of particles
    * @param bodyForces Volumetric forces acting on the particles
    * @param numberOfActiveParticles Number of active parameters
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