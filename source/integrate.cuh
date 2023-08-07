/**
 * @file integrate.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Numerical timestepping schemes
 * @date 2023.07.20.
 * 
 * Device side integrator algorithms
*/

#ifndef integrate_H
#define integrate_H

#include "particle.cuh"
#include "timestep.cuh"
#include "math.cuh"

namespace integrators
{
    __device__ inline var_type RK1(var_type dt, var_type x, var_type f)
    {
        return x + dt * f;
    }

    __device__ inline var_type AB2(var_type dt, var_type x, var_type f, var_type f_old)
    {
        return x + dt * (constant::AB2C1 * f - constant::AB2C2 * f_old);
    }

    __device__ inline var_type AM2(var_type dt, var_type x, var_type f, var_type f_old)
    {
        return x + dt * constant::AM2C1 * (f + f_old);
    }

    __device__ inline void euler(int tid, struct registerMemory &rmem, struct particle particles, struct timestepping timestep)
    {
        //velocity, v_i+1 = v_i + a*dt 
        rmem.v.x = RK1(timestep.dt,rmem.v.x,rmem.a[0].x);
        rmem.v.y = RK1(timestep.dt,rmem.v.y,rmem.a[0].y);
        rmem.v.z = RK1(timestep.dt,rmem.v.z,rmem.a[0].z);
        rmem.omega.x = RK1(timestep.dt,rmem.omega.x,rmem.beta[0].x);
        rmem.omega.y = RK1(timestep.dt,rmem.omega.y,rmem.beta[0].y);
        rmem.omega.z = RK1(timestep.dt,rmem.omega.z,rmem.beta[0].z);

        //position, x_i+1 = x_i + v*dt 
        rmem.u.x = RK1(timestep.dt,rmem.u.x,rmem.v.x);
        rmem.u.y = RK1(timestep.dt,rmem.u.y,rmem.v.y);
        rmem.u.z = RK1(timestep.dt,rmem.u.z,rmem.v.z);
    }

    __device__ inline void exact(int tid, struct registerMemory &rmem, struct particle particles, struct timestepping timestep)
    {
        //velocity, v_i+1 = v_i + a*dt 
        rmem.v.x = RK1(timestep.dt,rmem.v.x,rmem.a[0].x);
        rmem.v.y = RK1(timestep.dt,rmem.v.y,rmem.a[0].y);
        rmem.v.z = RK1(timestep.dt,rmem.v.z,rmem.a[0].z);
        rmem.omega.x = RK1(timestep.dt,rmem.omega.x,rmem.beta[0].x);
        rmem.omega.y = RK1(timestep.dt,rmem.omega.y,rmem.beta[0].y);
        rmem.omega.z = RK1(timestep.dt,rmem.omega.z,rmem.beta[0].z);

        //position, x_i+1 = x_i + v*dt + 0.5*a*t^2
        rmem.u.x = rmem.u.x + rmem.v.x * timestep.dt + constant::NUMBER_05 * rmem.a[0].x * timestep.dt * timestep.dt;
        rmem.u.y = rmem.u.y + rmem.v.y * timestep.dt + constant::NUMBER_05 * rmem.a[0].y * timestep.dt * timestep.dt;
        rmem.u.z = rmem.u.z + rmem.v.z * timestep.dt + constant::NUMBER_05 * rmem.a[0].z * timestep.dt * timestep.dt;
    }

    __device__ inline void adams2(int tid, struct registerMemory &rmem, struct particle particles, struct timestepping timestep, int step)
    {
        var_type velx,vely,velz;
        if(step == 0) 
        {
            // do an euler step
        rmem.v.x = RK1(timestep.dt,rmem.v.x,rmem.a[0].x);
        rmem.v.y = RK1(timestep.dt,rmem.v.y,rmem.a[0].y);
        rmem.v.z = RK1(timestep.dt,rmem.v.z,rmem.a[0].z);
        rmem.omega.x = RK1(timestep.dt,rmem.omega.x,rmem.beta[0].x);
        rmem.omega.y = RK1(timestep.dt,rmem.omega.y,rmem.beta[0].y);
        rmem.omega.z = RK1(timestep.dt,rmem.omega.z,rmem.beta[0].z);
        }
        else
        {
            //do an AB2 step to calculate velocity
            velx = AB2(timestep.dt,rmem.v.x,rmem.a[0].x,rmem.a[1].x);
            vely = AB2(timestep.dt,rmem.v.y,rmem.a[0].y,rmem.a[1].y);
            velz = AB2(timestep.dt,rmem.v.z,rmem.a[0].z,rmem.a[1].z);
            rmem.omega.x = AB2(timestep.dt,rmem.omega.x,rmem.beta[0].x,rmem.beta[1].x);
            rmem.omega.y = AB2(timestep.dt,rmem.omega.y,rmem.beta[0].y,rmem.beta[1].y);
            rmem.omega.z = AB2(timestep.dt,rmem.omega.z,rmem.beta[0].z,rmem.beta[1].z);
        }
        
        /*if(Debug == 2&& tid==0)
        {
            printf("a_old=(%6.3lf,%6.3lf,%6.3lf)\n",particles.a.x[tid+NumberOfParticles],particles.a.y[tid+NumberOfParticles],particles.a.z[tid+NumberOfParticles]);
            printf("a=(%6.3lf,%6.3lf,%6.3lf)\n",particles.a.x[tid],particles.a.y[tid],particles.a.z[tid]);
            printf("v_old=(%6.3lf,%6.3lf,%6.3lf)\n",particles.v.x[tid],particles.v.y[tid],particles.v.z[tid]);
            printf("v=(%6.3lf,%6.3lf,%6.3lf)\n",velx,vely,velz);
            printf("v=(%6.3lf,%6.3lf,%6.3lf)\n",velx,vely,velz);
        }*/

        //do an AM2 to calculate the position
        rmem.u.x = AM2(timestep.dt, rmem.u.x, rmem.v.x, velx);
        rmem.u.y = AM2(timestep.dt, rmem.u.y, rmem.v.y, vely);
        rmem.u.z = AM2(timestep.dt, rmem.u.z, rmem.v.z, velz);

        //save velocity
        rmem.v.x = velx;
        rmem.v.y = vely;
        rmem.v.z = velz;

        //save the old accelerations
        rmem.a[1].x = rmem.a[0].x;
        rmem.a[1].y = rmem.a[0].y;
        rmem.a[1].z = rmem.a[0].z;
        rmem.beta[1].x = rmem.beta[0].x;
        rmem.beta[1].y = rmem.beta[0].y;
        rmem.beta[1].z = rmem.beta[0].z;
        return;
    }


}


#endif