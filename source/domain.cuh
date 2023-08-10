/**
 * @file domain.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Description of the simulation domain
 * @date 2023.07.21.
 * 
 * Simulation domain (will be extended later)
*/

#ifndef domain_H
#define domain_H

#include "settings.cuh"
#include "timestep.cuh"
#include "math.cuh"

enum BoundaryConditionType {None, ReflectiveWall, HertzWall};
enum Direction {East, West, North, South, Top, Bottom};

/**
 * \brief Contains all the data about boundary conditions
*/
struct boundaryCondition
{
    ///Normal vector, pointing outwards
    vec3D n[NumberOfBoundaries];

    ///Point on the plane
    vec3D p[NumberOfBoundaries];

    //vector 1 of the traingle
    vec3D s[NumberOfBoundaries];

    //vector 2 of the traingle
    vec3D t[NumberOfBoundaries];

    //pre-calculated 1/|t|^2
    var_type t_scale[NumberOfBoundaries];

    //pre-calculated 1/|s|^2
    var_type s_scale[NumberOfBoundaries];

    ///Type of BC
    BoundaryConditionType type[NumberOfBoundaries];

    ///par1
    var_type alpha[NumberOfBoundaries];

    ///par2
    var_type beta[NumberOfBoundaries];

    ///par3
    var_type gamma[NumberOfBoundaries];

    ///parameter set for materials
    int material[NumberOfBoundaries];
};


namespace domainHandling
{
    void __device__ CalculateOverlap(int tid, struct registerMemory &rmem, int i, var_type d, struct contact &contacts)
    {
        //data
        contacts.tid[contacts.count] = -i-1; //assign negative id-s for these contacts
        contacts.deltan[contacts.count] = rmem.R - d;

        //check if they were contact in the last step
        bool wasInContact = false;
        for(int j = 0; j < MaxContactNumber; j++)
        {
            if(contacts.tid[contacts.count] == contacts.tid_last[j]) //they were in contact the last time
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
    }//end of calculateWallContact


    __device__ inline void applyBoundaryConditions(int tid, struct registerMemory &rmem, struct particle particles, struct boundaryCondition boundaryConditions, struct contact &contacts, struct materialParameters pars, struct timestepping timestep)
    {
        for(int i = 0; i < NumberOfBoundaries; i++)
        {
            //check for contact
            vec3D r(rmem.u.x,rmem.u.y,rmem.u.z);
            var_type d = -1.0f * (boundaryConditions.n[i] * (r - boundaryConditions.p[i]));

            if(d < rmem.R && d > -rmem.R) //particle and wall contact
            {
                    //chech validity of contact if STLs are used
                    bool contactValid = true;
                    if(domainType == DomainType::STL)
                    {   
                        vec3D q = r - boundaryConditions.n[i]*d - boundaryConditions.p[i];
                        var_type t = (q * boundaryConditions.t[i])*boundaryConditions.t_scale[i];
                        var_type s = (q * boundaryConditions.s[i])*boundaryConditions.s_scale[i];

                        /*if(tid == 0)
                        {
                            printf("d=%6.4lf  t=%6.4lf \t s=%6.4lf\n",d,t,s);
                        }*/


                        if(t < constant::ZERO || s < constant::ZERO || t + s > constant::NUMBER_1)
                        {
                            contactValid = false;
                        }
                    }

                    if(contactValid || domainType == DomainType::Rectangular)
                    {
                    //normal and tangentional velocity
                    vec3D v(rmem.v.x,rmem.v.y,rmem.v.z);
                    var_type vn_scalar = boundaryConditions.n[i] * v;
                    vec3D vn = boundaryConditions.n[i] * (vn_scalar);
                    vec3D vt = v - vn;

                    if(boundaryConditions.type[i] == BoundaryConditionType::ReflectiveWall)
                    {
                        //apply friction like velocity reduction
                        vec3D v_new;
                        v_new = v - vt * boundaryConditions.beta[i];

                        if(vn_scalar > 0) //flip sign of the normal velocity 
                        {
                            v_new = v_new - vn * (2.0f * boundaryConditions.alpha[i]) ;
                        }

                        rmem.v.x = v_new.x;
                        rmem.v.y = v_new.y;
                        rmem.v.z = v_new.z;

                        if(Debug == 2)
                        {
                            printf("tid=%d, v_n=%6.2lf v=(%6.2lf,%6.2lf,%6.2lf)\n",
                                tid,
                                vn_scalar,
                                v_new.x,v_new.y,v_new.z);
                        }


                        //put particle on the wall - NOT PHYSICAL, SHOULD BE FIXED AT SOME POINT
                        vec3D dr = boundaryConditions.n[i] * (d - rmem.R);
                        rmem.u.x += dr.x;
                        rmem.u.y += dr.y;
                        rmem.u.z += dr.z;


                    }//end of if reflective wall

                    if(boundaryConditions.type[i] == BoundaryConditionType::HertzWall)
                    {
                        //Reads last overlap
                        CalculateOverlap(tid,rmem,i,d,contacts);

                        //calculate new tangential overlap
                        contacts.deltat[contacts.count] = contacts.deltat[contacts.count] + (vt * timestep.dt);

                        //contact position of contact
                        contacts.p[i] = boundaryConditions.n[i]*d;
                        
                        //Stiffnesses
                        var_type Rdelta = sqrt(rmem.R*contacts.deltan[contacts.count]);
                        var_type Sn = constant::NUMBER_2 * pars.pairing[rmem.material].E_star[boundaryConditions.material[i]] * Rdelta;
                        var_type St = constant::NUMBER_8 * pars.pairing[rmem.material].G_star[boundaryConditions.material[i]] * Rdelta;

                        //normal elastic force
                        var_type Fne_scalar = constant::NUMBER_4o3 * pars.pairing[rmem.material].E_star[boundaryConditions.material[i]] * Rdelta * contacts.deltan[contacts.count];
                        vec3D Fne = boundaryConditions.n[i]* (-Fne_scalar);

                        //normal damping force
                        var_type Fnd_scalar = constant::DAMPING * pars.pairing[rmem.material].beta_star[boundaryConditions.material[i]] * sqrt(Sn * rmem.m);
                        vec3D Fnd = vn * Fnd_scalar;

                        //tangential elastic force
                        vec3D Fte;
                        Fte = contacts.deltat[contacts.count] * (-St);

                        //tangential damping force
                        var_type Ftd_scalar = constant::DAMPING * pars.pairing[rmem.material].beta_star[boundaryConditions.material[i]] * sqrt(St * rmem.m);
                        vec3D Ftd = vt * Ftd_scalar;

                        //total normal and tangentional force
                        vec3D Fn = Fne + Fnd;
                        vec3D Ft = Fte + Ftd;

                        //check for sliding
                        if(Ft.length() > Fn.length() * pars.pairing[rmem.material].mu0_star[boundaryConditions.material[i]])
                        {
                            //if sliding
                            Ft = Fn * pars.pairing[rmem.material].mu_star[boundaryConditions.material[i]];
                        }

                        //torque
                        vec3D M = contacts.p[i] ^ Ft;

                        //force
                        vec3D F = Fn + Ft;

                        //add the forces to the total
                        rmem.F.x += F.x;
                        rmem.F.y += F.y;
                        rmem.F.z += F.z;
                        rmem.M.x += M.x;
                        rmem.M.y += M.y;
                        rmem.M.z += M.z;

                        //check end
                        contacts.count++; 
                        if(contacts.count >= MaxContactNumber)
                        {
                            contacts.count = MaxContactNumber - 1;
                        }
                    }//end of Hertz Wall
                }//end of valid contact
            }//end of if contact
        }//end of for through boundaries
    }//end of function


}//end of namespace


#endif
