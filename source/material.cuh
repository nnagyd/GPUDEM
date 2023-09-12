/**
 * @file material.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Material description
 * @date 2023.09.12.
 * 
 * Struct to handle user given material parameters
*/

#ifndef material_H
#define material_H

/**
 * \brief Struct which contains the reduced quantities for material combinations
 */
struct materialContact
{
    var_type mu_star[NumberOfMaterials];
    var_type mu0_star[NumberOfMaterials];
    var_type mur_star[NumberOfMaterials];
    var_type E_star[NumberOfMaterials];
    var_type G_star[NumberOfMaterials];
    var_type beta_star[NumberOfMaterials];
};

/**
 * \brief Struct with all the user given material parameters, stored in the shared memory on the device side
 */
struct materialParameters
{
    ///Density
    var_type rho[NumberOfMaterials];
    ///Young's Modulus
    var_type E[NumberOfMaterials];
    ///Shear Modulus
    var_type G[NumberOfMaterials];
    ///Poisson ratio
    var_type nu[NumberOfMaterials];
    ///Restitution
    var_type e[NumberOfMaterials];
    ///Sliding friction coeff
    var_type mu[NumberOfMaterials];
    ///Static friction coeff
    var_type mu0[NumberOfMaterials];
    ///Rolling friction coeff
    var_type mur[NumberOfMaterials];
    ///Damping
    var_type beta[NumberOfMaterials];
    ///Lookup table for material pairinga
    struct materialContact pairing[NumberOfMaterials];
};


/**
 * \brief Contains all the function for material handling
 */
namespace materialHandling
{
    enum methods {Min, Max, HarmonicMean, Mean};

    /**
    * @brief Prints all the materials and material combinations
    * 
    * @param pars materialParameters struct with ALL material parameters
    * @param printPairings Settings to print the details of material pairings
    */
    void printMaterialInfo(struct materialParameters pars, bool printPairings = false)
    {   
        for(int i = 0; i < NumberOfMaterials; i++)
        {
            std::cout << "----- Material "<< i << " -----\n";
            std::cout << "            Density  [Rho]  = " <<  pars.rho[i] <<" kg/m3\n";
            std::cout << "       Young modulus E      = " <<  pars.E[i] <<" Pa \n";
            std::cout << "       Shear modulus G      = " <<  pars.G[i] <<" Pa \n";
            std::cout << "       Poisson ratio [nu]   = " <<  pars.nu[i] <<" \n";
            std::cout << "  Restitution coeff. e      = " <<  pars.e[i] <<" \n";
            std::cout << "     Friction coeff. [mu]   = " <<  pars.mu[i] <<" \n";
            std::cout << "Stat.friction coeff. [mu]_0 = " <<  pars.mu0[i] <<" \n";
            std::cout << "             Damping [beta] = " <<  pars.beta[i] <<" \n";

            if(printPairings)
            {
                for(int j = 0; j < NumberOfMaterials; j++)
                {
                    std::cout << "  --- Pairing with "<< j << " ---\n";
                    std::cout << "\t   Eq. Young modulus E*     = " <<  pars.pairing[i].E_star[j] <<" Pa \n";
                    std::cout << "\t   Eq. shear modulus G*     = " <<  pars.pairing[i].G_star[j] <<" Pa \n";
                    std::cout << "\t Eq. Friction coeff. [mu]   = " <<  pars.pairing[i].mu_star[j] <<" \n";
                    std::cout << "\tEq.stat.fric. coeff. [mu]_0 = " <<  pars.pairing[i].mu0_star[j] <<" \n";
                    std::cout << "\t         Eq. Damping [beta] = " <<  pars.pairing[i].beta_star[j] <<" \n";
                }
            }
        }
    }

    /**
    * @brief Calculates all the material pairings and damping
    * 
    * @param pars materialParameters struct with ALL material parameters
    * @param friction Method to calculate the friction coefficients
    * @param elastic Method to calculate the elastic coefficients
    * @param damping Method to calculate the damping coefficients
    */
    void calculateMaterialContact(struct materialParameters &pars, methods friction, methods elastic, methods damping)
    {
        //calculate beta
        for(int i = 0; i < NumberOfMaterials; i++)
        {
            var_type loge = log(pars.e[i]);
            pars.beta[i] = -loge / sqrt(loge*loge + constant::PI*constant::PI);
        }

        for(int i = 0; i < NumberOfMaterials; i++)
        {
            for(int j = 0; j < NumberOfMaterials; j++)
            {
                //------ calculate friction ------
                if(friction == methods::Min)
                {
                    pars.pairing[i].mu_star[j] = min(pars.mu[i],pars.mu[j]);
                    pars.pairing[i].mu0_star[j] = min(pars.mu0[i],pars.mu0[j]);
                    pars.pairing[i].mur_star[j] = min(pars.mur[i],pars.mur[j]);
                }
                if(friction == methods::Max)
                {
                    pars.pairing[i].mu_star[j] = max(pars.mu[i],pars.mu[j]);
                    pars.pairing[i].mu0_star[j] = max(pars.mu0[i],pars.mu0[j]);
                    pars.pairing[i].mur_star[j] = max(pars.mur[i],pars.mur[j]);
                }                
                if(friction == methods::Mean)
                {
                    pars.pairing[i].mu_star[j] = constant::NUMBER_05*(pars.mu[i]+pars.mu[j]);
                    pars.pairing[i].mu0_star[j] = constant::NUMBER_05*(pars.mu0[i]+pars.mu0[j]);
                    pars.pairing[i].mur_star[j] = constant::NUMBER_05*(pars.mur[i]+pars.mur[j]);
                }

                //------ calculate elastic parameters ------ 
                if(elastic == methods::Min)
                {
                    pars.pairing[i].E_star[j] = min(pars.E[i],pars.E[j]);
                    pars.pairing[i].G_star[j] = min(pars.G[i],pars.G[j]);
                }
                if(elastic == methods::Max)
                {
                    pars.pairing[i].E_star[j] = max(pars.E[i],pars.E[j]);
                    pars.pairing[i].G_star[j] = max(pars.G[i],pars.G[j]);
                }
                if(elastic == methods::Mean)
                {
                    pars.pairing[i].E_star[j] = constant::NUMBER_05*(pars.E[i] + pars.E[j]);
                    pars.pairing[i].G_star[j] = constant::NUMBER_05*(pars.G[i] + pars.G[j]);
                }
                if(elastic == methods::HarmonicMean)
                {
                    var_type div1 = constant::NUMBER_1 - pars.nu[i]*pars.nu[i];
                    var_type div2 = constant::NUMBER_1 - pars.nu[j]*pars.nu[j];

                    pars.pairing[i].E_star[j] = constant::NUMBER_1/(div1/pars.E[i] + div2/pars.E[j]);

                    var_type nu_star = constant::NUMBER_2/( constant::NUMBER_1/pars.nu[i] + constant::NUMBER_1/pars.nu[j]  );

                    pars.pairing[i].G_star[j] = constant::NUMBER_05 * pars.pairing[i].E_star[j] / (constant::NUMBER_1 + nu_star);
                }

                //------ calculate beta ------ 
                if(damping == methods::Min)
                {
                    pars.pairing[i].beta_star[j] = min(pars.beta[i],pars.beta[j]);
                }
                if(damping == methods::Max)
                {
                    pars.pairing[i].beta_star[j] = max(pars.beta[i],pars.beta[j]);
                }
                if(damping == methods::Mean)
                {
                    pars.pairing[i].beta_star[j] = constant::NUMBER_05*(pars.beta[i]+pars.beta[j]);
                }
                if(damping == methods::HarmonicMean)
                {
                    pars.pairing[i].beta_star[j] = constant::NUMBER_2/(constant::NUMBER_1/pars.beta[i] + constant::NUMBER_1/pars.beta[j]);
                }



            }
        }
    }
}

#endif