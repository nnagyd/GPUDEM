
/**
 * @file randomgen.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Random number generation 
 * @date 2023.09.12.
 * 
 * Random number generations implemented
*/

#ifndef random_H
#define random_H

 #include <cstdlib>
 #include <ctime>

 #include "math.cuh"

 /**
 * \brief Contains everything necessary for random generation
 */
 namespace RandomGeneration
 {
    var_type overRandMax = constant::NUMBER_1/var_type(RAND_MAX);

    /**
    * \brief Initializes a random seed based on time
    */
    void initializeRandomSeed()
    {
        srand(time(NULL));
    }

    /**
    * \brief Initializes a random seed based on a given number
    *
    * @param seed seed of srand()
    */
    void initializeRandomSeed(int seed)
    {
        srand(seed);
    }

    /**
    * \brief Generates a random var_type in a range
    * 
    * @param min lower end of the range
    * @param max higher end of the range
    */
    var_type randomInRange(var_type min, var_type max)
    {
        var_type x = (var_type(rand())*overRandMax)*(max-min) + min;
        return x;
    }

 }


#endif