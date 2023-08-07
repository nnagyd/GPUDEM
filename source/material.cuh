/**
 * @file material.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Material description
 * @date 2023.07.21.
 * 
 * Struct to handle user given material parameters
*/

#ifndef material_H
#define material_H

/**
 * \brief Struct with all the user given material parameters, stored in the shared memory on the device side
 *
 */
struct materialParameters
{
    ///Density
    var_type rho;
    ///Young's Modulus
    var_type E;
    ///Shear Modulus
    var_type G;
    ///Poisson ratio
    var_type nu;
    ///Damping factor
    var_type beta;
    ///Sliding friction coeff
    var_type mu;
    ///Rolling friction coeff
    var_type mu0;
};







#endif