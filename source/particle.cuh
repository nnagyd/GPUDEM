/**
 * @file particle.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Particle and particle cloud discriptions 
 * @date 2023.09.12.
 * 
 * Device and host side instance of a particle
*/

#ifndef particle_H
#define particle_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "settings.cuh"
#include "randomgen.cuh"
#include "material.cuh"
#include "math.cuh"



/**
    \brief Cartesian coordinates
*/
struct coordinate
{
    /// x coodinate
    var_type x;
    /// y coodinate
    var_type y;
    /// z coodinate
    var_type z;
};

/**
    \brief Cartesian coordinate vectors
*/
struct coordinates
{
    /// x coodinate
    var_type *x;
    /// y coodinate
    var_type *y;
    /// z coodinate
    var_type *z;
};


/**
    \brief Coordinates, velocity and radius of a particles
*/
struct particle
{
    /// position
    struct coordinates u;

    /// velocity
    struct coordinates v;

    ///acceleration
    struct coordinates a;

    ///angular velocity
    struct coordinates omega;

    ///angular acceleration
    struct coordinates beta;

    ///force acting on the particle
    struct coordinates F;

    ///torque acting on the particle
    struct coordinates M;

    /// Radius of the particle 
    var_type *R; 

    /// Mass of the particle
    var_type *m;

    /// Inertia of the particle
    var_type *theta;

    ///Inverse of radius of the particle 
    var_type *R_rec; 

    /// Inverse of mass of the particle
    var_type *m_rec;

    /// Inverse of inertia of the particle
    var_type *theta_rec;

    /// Material set
    int *material;

    ///Particle cell id
    int *cid;

    ///Stores the linked cell list
    int * linkedCellList;

    ///Number of particles in cell
    int * NinCell;
};

/**
    \brief All information necessary to generate the initial particles
*/
struct particleDistribution
{
    /// Min. coord. values
    struct coordinate min;
    /// Max. coord. values
    struct coordinate max;
    /// Average velocity
    var_type vmean;
    /// Standard deviation of velocity
    var_type vsigma;
    ///Mean radius
    var_type Rmean;
    ///Standard deviation of the radius
    var_type Rsigma;
};


/**
 * \brief Contains all the functions and structs necessary for handling the particles
 */
namespace particleHandling
{
    enum class ParticleSizeDistribution { None, Uniform, Gauss };
    enum class ParticleVelocityDistribution { None, Uniform, Gauss };

    /**
    * \brief Fills up the particle struct p from a given material data
    *
    * @param p Pre-allocated memory where the particle data is saved
    * @param pars Physical parameters given by the user
    * @param mat_id Material ID 
    * @param start_id Starting index of this material 
    * @param end_id End index of this material 
    *
    * @return ensamble of particles according to the distribution described in pdist
    */
    void generateParticleParameters(struct particle p, struct materialParameters pars, int mat_id, int start_id, int end_id)
    {
        for(int i = start_id; i < end_id; i++)
        {
            //mass and inertia
            p.m[i] = pars.rho[mat_id] * p.R[i] * p.R[i] * p.R[i] * constant::VOLUME_FACTOR;
            p.theta[i] = constant::NUMBER_04 * p.m[i] * p.R[i] * p.R[i];

            //Reciprocals
            p.R_rec[i] = constant::NUMBER_1/p.R[i];
            p.m_rec[i] = constant::NUMBER_1/p.m[i];
            p.theta_rec[i] = constant::NUMBER_1/p.theta[i];

            //physical properties
            p.material[i] = mat_id;
        }
    }


    /**
    * \brief Particle generation based on the initial particle distribution
    *
    * @param particles Pre-allocated memory where the particle data is saved
    * @param pdist Particle distribution information based on the particleDistribution struct
    * @param psize_dist Particle size distribuation chosen from the ParticleSizeDistribution enum
    * @param pvel_dist Particle velocity distribuation chosen from the ParticleVelocityDistribution enum
    *
    * @return ensamble of particles according to the distribution described in pdist
    */
    void generateParticleLocation(struct particle p, struct particleDistribution pdist, ParticleSizeDistribution psize_dist, ParticleVelocityDistribution pvel_dist)
    {
        for(int i = 0; i < NumberOfParticles; i++)
        {
            //radius
            if(psize_dist == ParticleSizeDistribution::None)
            {
                p.R[i] = pdist.Rmean;
            }
            if(psize_dist == ParticleSizeDistribution::Uniform)
            {
                p.R[i] = RandomGeneration::randomInRange(pdist.Rmean - pdist.Rsigma, pdist.Rmean + pdist.Rsigma);
            }
            if(psize_dist == ParticleSizeDistribution::Gauss)
            {
                //not implemented yet
            }
        
            //position
            p.u.x[i] = RandomGeneration::randomInRange(pdist.min.x+p.R[i], pdist.max.x-p.R[i]);
            p.u.y[i] = RandomGeneration::randomInRange(pdist.min.y+p.R[i], pdist.max.y-p.R[i]);
            p.u.z[i] = RandomGeneration::randomInRange(pdist.min.z+p.R[i], pdist.max.z-p.R[i]);
        
            //velocity
            if(pvel_dist == ParticleVelocityDistribution::None)
            {
                p.v.x[i] = pdist.vmean;
                p.v.y[i] = pdist.vmean;
                p.v.z[i] = pdist.vmean;
            }
            if(pvel_dist == ParticleVelocityDistribution::Uniform)
            {
                p.v.x[i] = RandomGeneration::randomInRange(pdist.vmean - pdist.vsigma, pdist.vmean + pdist.vsigma);
                p.v.y[i] = RandomGeneration::randomInRange(pdist.vmean - pdist.vsigma, pdist.vmean + pdist.vsigma);
                p.v.z[i] = RandomGeneration::randomInRange(pdist.vmean - pdist.vsigma, pdist.vmean + pdist.vsigma);
            }
        }
    }


    /**
    * \brief Particle generation based on coordinate lists
    *
    * @param particles Pre-allocated memory where the particle data is saved
    * @param coords Coordinates of the particles
    * @param radii Radii of the particles
    * @param pars  Physical parameters given by the user
    *
    * @return ensamble of particles according to the distribution described in pdist
    */
    void generateParticleLocation(struct particle p, struct coordinates coords, var_type * radii, struct materialParameters pars)
    {
        for(int i = 0; i < NumberOfParticles; i++)
        {
            //radius
            p.R[i] = radii[i];

            //position
            p.u.x[i] = coords.x[i];
            p.u.y[i] = coords.y[i];
            p.u.z[i] = coords.z[i];
        }
    }


    /**
    * \brief Print the data of a list of particles
    *
    * @param particles A list of particles
    */
    void printParticles(struct particle p)
    {
        for(int i = 0; i < NumberOfParticles; i++)
        {
            std::cout <<"p="<< std::setw(9) << p.u.x[i] << "," << std::setw(9) << p.u.y[i] << "," << std::setw(9) << p.u.z[i];
            std::cout <<"\tF="<< std::setw(9) << p.F.x[i] << std::setw(9) << p.F.y[i] << std::setw(9) << p.F.z[i];
            std::cout <<"\tM="<< std::setw(9) << p.M.x[i] << std::setw(9) << p.M.y[i] << std::setw(9) << p.M.z[i] << "\n";
        }
    }

}//end of namespace

#endif