/**
 * @file domain.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Description of the simulation domain
 * @date 2023.09.12.
 * 
 * Contains the boundary description (STL or Rectangular) and the boundary 
 * condition calculations
*/

#ifndef domain_H
#define domain_H

#include "settings.cuh"
#include "timestep.cuh"
#include "math.cuh"


enum BoundaryConditionType {None, ReflectiveWall, HertzWall};

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

    //force acting on the triangle
    vec3D *F;

    //torque acting on the triangle
    vec3D *M;

    ///Type of BC
    BoundaryConditionType type[NumberOfBoundaries];

    /*///par1
    var_type alpha[NumberOfBoundaries];

    ///par2
    var_type beta[NumberOfBoundaries];*/

    ///parameter set for materials
    int material[NumberOfBoundaries];
};

/**
 * \brief Handling of boundary conditions and STL files
*/
namespace domainHandling
{
    /**
    * @brief Calculates the contact parameters between a particle and a boundary
    * 
    * @param tid Thread index of the particle
    * @param rmem Register memory containing all the data about the particle
    * @param i Index of the domain boundary, particle tid is in contact with
    * @param d Distance between particle and domain boundary
    * @param contacts List of contacts
    * 
    * @return the contact struct is filled up 
    */
    __device__ inline void CalculateOverlap(int tid, struct registerMemory &rmem, int i, var_type d, struct contact &contacts)
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


    /**
    * @brief Calculates the forces based on the boundary constraints and given model
    * 
    * @param tid Thread index of the particle
    * @param rmem Register memory containing all the data about the particle
    * @param particles List of all particle data
    * @param boundaryConditions List of all boundary condition data
    * @param contacts List of all contact data
    * @param pars All material parameters
    * @param timestep Timestep specificiations
    *
    * 
    */
    __device__ inline void applyBoundaryConditions(int tid, struct registerMemory &rmem, struct particle particles, struct boundaryCondition &boundaryConditions, struct contact &contacts, struct materialParameters pars, struct timestepping timestep)
    {
        for(int i = 0; i < NumberOfBoundaries; i++)
        {
            //check for contact
            vec3D r(rmem.u.x,rmem.u.y,rmem.u.z);
            vec3D rmp = r - boundaryConditions.p[i];
            var_type d = -1.0f * (boundaryConditions.n[i] * rmp);

            if(d < rmem.R && d > -rmem.R) //particle and wall contact
            {
                //chech validity of contact if STLs are used
                bool contactValid = true;
                if(domainType == DomainType::STL)
                {  
                    var_type t = boundaryConditions.t[i]*rmp;
                    var_type s = boundaryConditions.s[i]*rmp;

                    if(t < constant::ZERO || s < constant::ZERO || t + s > constant::NUMBER_1)
                    {
                        contactValid = false;
                    }
                }

                if(contactValid)
                {
                    //normal and tangentional velocity
                    vec3D v(rmem.v.x,rmem.v.y,rmem.v.z);
                    vec3D omega(rmem.omega.x,rmem.omega.y,rmem.omega.z);
                    vec3D v_rel = v + ((omega *rmem.R) ^ boundaryConditions.n[i]);
                    var_type vn_rel_norm = boundaryConditions.n[i] * v_rel;
                    vec3D vn_rel = boundaryConditions.n[i] * (vn_rel_norm);
                    vec3D vt_rel = v_rel - vn_rel;

                    /*if(boundaryConditions.type[i] == BoundaryConditionType::ReflectiveWall)
                    {
                        //apply friction like velocity reduction
                        vec3D v_new;
                        v_new = v - vt_rel * boundaryConditions.beta[i];

                        if(vn_rel_norm > 0) //flip sign of the normal velocity 
                        {
                            v_new = v_new - vn_rel * (2.0f * boundaryConditions.alpha[i]) ;
                        }

                        rmem.v.x = v_new.x;
                        rmem.v.y = v_new.y;
                        rmem.v.z = v_new.z;

                        //put particle on the wall - NOT PHYSICAL, SHOULD BE FIXED AT SOME POINT
                        vec3D dr = boundaryConditions.n[i] * (d - rmem.R);
                        rmem.u.x += dr.x;
                        rmem.u.y += dr.y;
                        rmem.u.z += dr.z;


                    }//end of if reflective wall*/

                    if(boundaryConditions.type[i] == BoundaryConditionType::HertzWall)
                    {
                        //Reads last overlap
                        CalculateOverlap(tid,rmem,i,d,contacts);

                        //calculate new tangential overlap
                        contacts.deltat[contacts.count] = contacts.deltat[contacts.count] + (vt_rel * timestep.dt);

                        //contact position of contact
                        contacts.p[contacts.count] = boundaryConditions.n[i]*d;
                        
                        //Stiffnesses
                        var_type Rdelta = sqrt(rmem.R*contacts.deltan[contacts.count]);
                        var_type Sn = constant::NUMBER_2 * pars.pairing[rmem.material].E_star[boundaryConditions.material[i]] * Rdelta;
                        var_type St = constant::NUMBER_8 * pars.pairing[rmem.material].G_star[boundaryConditions.material[i]] * Rdelta;

                        //normal elastic force
                        var_type Fne_norm = constant::NUMBER_4o3 * pars.pairing[rmem.material].E_star[boundaryConditions.material[i]] * Rdelta * contacts.deltan[contacts.count];
                        vec3D Fne = boundaryConditions.n[i]* (-Fne_norm);

                        //normal damping force
                        var_type Fnd_norm = constant::DAMPING * pars.pairing[rmem.material].beta_star[boundaryConditions.material[i]] * sqrt(Sn * rmem.m);
                        vec3D Fnd = vn_rel * Fnd_norm;

                        //tangential elastic force
                        vec3D Fte;
                        Fte = contacts.deltat[contacts.count] * (-St);

                        //tangential damping force
                        var_type Ftd_norm = constant::DAMPING * pars.pairing[rmem.material].beta_star[boundaryConditions.material[i]] * sqrt(St * rmem.m);
                        vec3D Ftd = vt_rel * Ftd_norm;

                        //total normal and tangentional force
                        vec3D Fn = Fne + Fnd;
                        vec3D Ft = Fte + Ftd;

                        //check for sliding
                        var_type Ft_norm = Ft.length();
                        var_type Fn_norm = Fn.length();
                        if(Ft_norm > Fn_norm  * pars.pairing[rmem.material].mu0_star[boundaryConditions.material[i]])
                        {
                            //if sliding
                            Ft = Ft*(Fn_norm/Ft_norm * pars.pairing[rmem.material].mu_star[boundaryConditions.material[i]]);
                        }

                        //torque
                        vec3D M = contacts.p[contacts.count] ^ Ft;

                        //calculate rolling
                        if(RollingFriction)
                        {
                            var_type omega_norm = omega.length();
                            if(omega_norm != constant::ZERO)
                            {
                                vec3D omega_unit = omega * (constant::NUMBER_1 / omega_norm);
                                vec3D Mr = omega_unit * (-pars.pairing[rmem.material].mur_star[boundaryConditions.material[i]] * Fn_norm * contacts.p[contacts.count].length());
                                M = M + Mr;
                            }
                        }

                        //force
                        vec3D F = Fn + Ft;

                        //add the forces to the total
                        rmem.F.x += F.x;
                        rmem.F.y += F.y;
                        rmem.F.z += F.z;
                        rmem.M.x += M.x;
                        rmem.M.y += M.y;
                        rmem.M.z += M.z;

                        if(SaveForcesTriangles)
                        {
                            //save the force acting on the boundary
                            boundaryConditions.F[i].x += F.x;
                            boundaryConditions.F[i].y += F.y;
                            boundaryConditions.F[i].z += F.z;
                            boundaryConditions.M[i].x += M.x;
                            boundaryConditions.M[i].y += M.y;
                            boundaryConditions.M[i].z += M.z;
                        }

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


    /**
    * @brief Prints the boundary conditions
    * 
    * @param BC List of all boundary condition data
    * @param printGeometry Prints geometric data
    * @param printMaterial Prints material data
    */
    void printBoundaryConditions(struct boundaryCondition BC, bool printGeometry = true, bool printMaterial = true)
    {
        for(int i = 0; i < NumberOfBoundaries; i++)
        {
            if(printGeometry)
            {
                std::cout << "----- BC " << i << " ----- \n";
                std::cout << "      p = (" << BC.p[i].x << "," << BC.p[i].y << "," << BC.p[i].z << ")\n"; 
                std::cout << "      n = (" << BC.n[i].x << "," << BC.n[i].y << "," << BC.n[i].z << ")\n"; 
                std::cout << "      s = (" << BC.s[i].x << "," << BC.s[i].y << "," << BC.s[i].z << ")\n"; 
                std::cout << "      t = (" << BC.t[i].x << "," << BC.t[i].y << "," << BC.t[i].z << ")\n"; 
            }
            if(printMaterial)
            {
                std::cout << "   type = ";
                switch(BC.type[i])
                {
                    case None: std::cout << "None\n"; break;
                    /*case ReflectiveWall: 
                        std::cout << "ReflectiveWall\n"; 
                        std::cout << "   beta = " << BC.beta[i]<< "\n";
                        std::cout << "  alpha = " << BC.alpha[i]<< "\n";
                    break;*/
                    case HertzWall: 
                        std::cout << "HertzWall\n"; 
                        std::cout << "material= " << BC.material[i] << "\n";
                    break;
                }
            }
        }//end of for
    }//end of print

    /**
    * @brief Translates the boundary conditions and flips normal vectors
    * 
    * @param BC List of all boundary condition data
    * @param startId startId in the BC struct of the boundaries
    * @param printMaterial Prints material data
    */
    void translateBoundaryConditions(struct boundaryCondition &BC, int startId, int endId, var_type x, var_type y, var_type z, bool flipNormals = false)
    {
        for(int i = startId; i < endId; i++)
        {
            BC.p[i].x += x;
            BC.p[i].y += y;
            BC.p[i].z += z;

            if(flipNormals)
            {
                BC.n[i].x *= - constant::NUMBER_1;
                BC.n[i].y *= - constant::NUMBER_1;
                BC.n[i].z *= - constant::NUMBER_1;
            }
        }//end of for
    }//end of print


    /**
    * @brief Prints the boundary conditions
    * 
    * @param BCsH BCs on the host side
    * @param BCsD BCs on the device side
    */
    void convertBoundaryConditions(struct boundaryCondition BCsH, struct boundaryCondition &BCsD)
    {
        for(int i = 0; i < NumberOfBoundaries; i++)
        {
            var_type detBeta = calculateDetBeta(BCsH.s[i],BCsH.t[i],BCsH.n[i]);
            var_type overDetBeta = constant::NUMBER_1 / detBeta;
            //vector n
            BCsD.n[i] = BCsH.n[i];

            //vector p
            BCsD.p[i] = BCsH.p[i];

            //vector s
            BCsD.s[i].x = overDetBeta * (BCsH.n[i].z * BCsH.t[i].y - BCsH.n[i].y * BCsH.t[i].z );
            BCsD.s[i].y = overDetBeta * (BCsH.n[i].x * BCsH.t[i].z - BCsH.n[i].z * BCsH.t[i].x );
            BCsD.s[i].z = overDetBeta * (BCsH.n[i].y * BCsH.t[i].x - BCsH.n[i].x * BCsH.t[i].y );

            //vector t
            BCsD.t[i].x = overDetBeta * (BCsH.n[i].y * BCsH.s[i].z - BCsH.n[i].z * BCsH.s[i].y );
            BCsD.t[i].y = overDetBeta * (BCsH.n[i].z * BCsH.s[i].x - BCsH.n[i].x * BCsH.s[i].z );
            BCsD.t[i].z = overDetBeta * (BCsH.n[i].x * BCsH.s[i].y - BCsH.n[i].y * BCsH.s[i].x );

            //other
            BCsD.type[i] = BCsH.type[i];
            /*BCsD.alpha[i] = BCsH.alpha[i];
            BCsD.beta[i] = BCsH.beta[i];*/
            BCsD.material[i] = BCsH.material[i];
            
        }//end of for
    }//end of convert

}//end of namespace


#endif
