/**
 * @file timestep.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Timestepping settings
 * @date 2023.07.20.
 * 
 * Device side integrator algorithms
*/

#ifndef timestep_H
#define timestep_H

#include <iostream>

/**
 * \brief Timestep settings
*/
struct timestepping
{
    ///Start time of the simulation
    var_type starttime;
    ///End time of the simulation
    var_type endtime;
    ///Timestep size
    var_type dt;
    ///Frequency of saves
    var_type savetime;
    ///Number of steps in the simulation
    int numberOfSteps;
    ///Number of steps between saves
    int saveSteps;

    ///initialize based on the number of steps between saves
    timestepping(var_type start, var_type end, var_type dt, int saveSteps):starttime(start), endtime(end), dt(dt), saveSteps(saveSteps)
    {
        this->numberOfSteps = int((end - start) / dt) + 1;
        this->savetime = dt * saveSteps;
    }

    ///initialize based on the time between saves
    timestepping(var_type start, var_type end, var_type dt, var_type savetime):starttime(start), endtime(end), dt(dt), savetime(savetime)
    {
        this->numberOfSteps = int((end - start) / dt) + 1;
        this->saveSteps = int((savetime - start) / dt) + 1;
    }
};

/**
 * \brief Functions for handling time
*/
namespace timeHandling
{

    /**
     * @brief Prints the timestep settings
     * 
     * @param timestep The timestepping struct containg all data about the timesteps
     */
    void printTimestepSettings(struct timestepping timestep)
    {
        std::cout << " Start: " << timestep.starttime << "\n";
        std::cout << "   End: " << timestep.endtime << "\n";
        std::cout << "  Save: " << timestep.savetime << "\n";
        std::cout << "    Dt: " << timestep.dt << "\n";
        std::cout << "#steps:" << timestep.numberOfSteps << "\n";
        std::cout << "#saves:" << timestep.saveSteps << "\n";
    }
}

#endif