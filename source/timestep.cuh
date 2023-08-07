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

struct timestepping
{
    var_type starttime;
    var_type endtime;
    var_type dt;
    var_type savetime;
    int numberOfSteps;
    int saveSteps;

    timestepping(var_type start, var_type end, var_type dt, int saveSteps):starttime(start), endtime(end), dt(dt), saveSteps(saveSteps)
    {
        this->numberOfSteps = int((end - start) / dt) + 1;
        this->savetime = dt * saveSteps;
    }

    timestepping(var_type start, var_type end, var_type dt, var_type savetime):starttime(start), endtime(end), dt(dt), savetime(savetime)
    {
        this->numberOfSteps = int((end - start) / dt) + 1;
        this->saveSteps = int((savetime - start) / dt) + 1;
    }
};

namespace timeHandling
{
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