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
#include "math.cuh"

enum BoundaryConditionType {None, ReflectiveWall};
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

    ///Type of BC
    BoundaryConditionType type[NumberOfBoundaries];

    ///par1
    var_type alpha[NumberOfBoundaries];

    ///par2
    var_type beta[NumberOfBoundaries];

    ///par3
    var_type gamma[NumberOfBoundaries];
};


namespace domainHandling
{
    __device__ inline void applyBoundaryConditions(int tid, struct registerMemory &rmem, struct particle particles, struct boundaryCondition boundaryConditions)
    {
        for(int i = 0; i < NumberOfBoundaries; i++)
        {
            //check for contact
            vec3D r(rmem.u.x,rmem.u.y,rmem.u.z);
            var_type d = -1.0f * (boundaryConditions.n[i] * (r - boundaryConditions.p[i]));

            if(d < rmem.R) //particle and wall contact
            {
                if(Debug == 2)
                {
                    printf("tid=%d, d=%6.2lf\n",tid,d);
                }
                
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
            }//end of if contact
        }//end of for through boundaries
    }//end of function


}//end of namespace


#endif
