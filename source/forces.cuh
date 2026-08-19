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
            //particle index of the i.th contact
            int tidi = contacts.tid[i];

            //read the other particles data into the registers
            if(tidi >= 0)
            {
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
                var_type Rdelta = sqrt(contacts.Rstar[i]*abs(contacts.deltan[i]));
                var_type Sn = constant::NUMBER_2 * pars.pairing[rmem.material].E_star[contacts.material[i]] * Rdelta;
                var_type St = constant::NUMBER_8 * pars.pairing[rmem.material].G_star[contacts.material[i]] * Rdelta;

                /// FORCES
                //normal elastic force
                var_type Fne_norm = constant::NUMBER_4o3 * pars.pairing[rmem.material].E_star[contacts.material[i]] * Rdelta * contacts.deltan[i];
                if(AdhesionForce)
                {
                    //modify the normal force with the adhesion (JKR theory)
                    var_type mod = sqrt(constant::NUMBER_16*constant::PI*pars.sigma*pars.pairing[rmem.material].E_star[contacts.material[i]]*Rdelta) * Rdelta;
                    Fne_norm -= mod;
                }

                //normal damping force
                var_type Fnd_norm = constant::DAMPING * pars.pairing[rmem.material].beta_star[contacts.material[i]] * sqrt(Sn * contacts.mstar[i]);

                //tangential damping force
                var_type Ftd_norm = constant::DAMPING * pars.pairing[rmem.material].beta_star[contacts.material[i]] * sqrt(St * contacts.mstar[i]);

                if(WaterBridges && contacts.deltan[i] < constant::ZERO) //if the particles are not in contact, but might have a water bridge
                {
                    //modify the normal force with the effect of water bridges
                    var_type Vls = constant::NUMBER_4 * constant::PI * pars.psi * rmem.R * rmem.R;
                    var_type Vli = constant::NUMBER_4 * constant::PI * pars.psi * Ri * Ri;
                    var_type Ns = 8.0f;

                    var_type Vl = (Vli + Vls) / Ns;
                    var_type zeta = (constant::NUMBER_1 + constant::NUMBER_2 * Vl / (constant::PI * contacts.Rstar[i] * contacts.deltan[i] * contacts.deltan[i])) - constant::NUMBER_1;
                    var_type mod = constant::NUMBER_4 * constant::PI * contacts.Rstar[i] * pars.sigma * cos(pars.pairing[rmem.material].theta_star[contacts.material[i]]) / (constant::NUMBER_1 + constant::NUMBER_1 / zeta);

                    if(-contacts.deltan[i] < pars.psi) //valid water bridge
                    {
                        Fne_norm = -mod;
                    }
                    else
                    {
                        Fne_norm = constant::ZERO;
                    }

                    //printf("Water bridge particle %d, Fne = %lf\n",tid,Fne_norm);

                    Fnd_norm = constant::ZERO;
                    contacts.deltat[i].x = constant::ZERO;
                    contacts.deltat[i].y = constant::ZERO;
                    contacts.deltat[i].z = constant::ZERO;
                    Ftd_norm = constant::ZERO;
                }

                //force vectors
                vec3D Fne = contacts.r[i]* (-Fne_norm);
                vec3D Fnd = vn_rel * Fnd_norm;
                vec3D Fte = contacts.deltat[i] * (-St); //tangential elastic force
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

                /*printf("Fne=(%lf,%lf,%lf)\n",Fne.x,Fne.y,Fne.z);
                printf("Fnd=(%lf,%lf,%lf)\n",Fnd.x,Fnd.y,Fnd.z);
                printf("Fte=(%lf,%lf,%lf)\n",Fte.x,Fte.y,Fte.z);
                printf("Ftd=(%lf,%lf,%lf)\n",Ftd.x,Ftd.y,Ftd.z);
                printf("dt=(%lf,%lf,%lf)\n", contacts.deltat[i].x, contacts.deltat[i].y, contacts.deltat[i].z);
                printf("F=(%lf,%lf,%lf)\n",F.x,F.y,F.z);
                while(true)
                {
                    1;
                }*/

                //add the force and torque to the total
                rmem.F.x += F.x;
                rmem.F.y += F.y;
                rmem.F.z += F.z;
                rmem.M.x += M.x;
                rmem.M.y += M.y;
                rmem.M.z += M.z;

                //calculate the stresses
                //shear
                particles.F.z[tid] += F.z * contacts.p[i].z;
                particles.F.x[tid] += F.x * contacts.p[i].z;
                particles.F.y[tid] += F.y * contacts.p[i].z;
            }
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
            if(!isnan(particles.u.x[i]))
            {
                k += constant::NUMBER_05*particles.m[i]*(particles.v.x[i]*particles.v.x[i] + particles.v.y[i]*particles.v.y[i] + particles.v.z[i]*particles.v.z[i]) + constant::NUMBER_05*particles.theta[i]*(particles.omega.x[i]*particles.omega.x[i] + particles.omega.y[i]*particles.omega.y[i] + particles.omega.z[i]*particles.omega.z[i]);
            }
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
            if(!isnan(particles.u.x[i]))
            {
                p += -particles.m[i]*(particles.u.x[i]*bodyForces.x + particles.u.y[i]*bodyForces.y + particles.u.z[i]*bodyForces.z);
            }
        }
        return p;
    }
}

#endif