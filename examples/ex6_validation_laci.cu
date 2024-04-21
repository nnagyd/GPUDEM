/**
 * @file ex1_deposition.cu
 * @author Dániel NAGY
 * @version 1.0
 * @brief Gravitational deposition example
 * @date 2023.09.12.
 * 
 * Validation with EDEM
*/


#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>

constexpr int NumberOfParticles = 2;
constexpr int NumberOfMaterials = 1;
constexpr int NumberOfBoundaries = 0;

constexpr int numberOfActiveParticles = 2;

#include "source/solver.cuh"

int main(int argc, char const *argv[])
{
    //Set GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    //material parameters
    struct materialParameters materials;
    materials.rho[0]= 2500.0f;
    materials.E[0] = 2.5e8f;
    materials.G[0] = 1.0e8f;
    materials.nu[0] = 0.25f;
    materials.e[0] = 0.5f;
    materials.mu[0] = 100.0f;
    materials.mu0[0] = 100.0f;

    materialHandling::calculateMaterialContact(materials,
        materialHandling::methods::Min, //friction
        materialHandling::methods::HarmonicMean, //E, G
        materialHandling::methods::Min); //damping
    materials.pairing[0].G_star[0] = materials.G[0];


    //timestep settings
    float dt = 0.0001f;
    float end = 0.1589f;
    struct timestepping timestep(0.0f,end,dt,end);

    //body forces
    struct bodyForce gravity;
    gravity.x = 0.0f;
    gravity.y = 0.0f;
    gravity.z = 0.0f;

    //BCs - wall on the bottom at z=0 and around in a 2m x 2m domain
    struct boundaryCondition BCs;

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    ioHandling::readParticlesCSV(particlesH,"data/ex6_2.dat",numberOfActiveParticles);
    particleHandling::generateParticleParameters(particlesH,materials,0,0,numberOfActiveParticles);
    particlesH.v.z[1] = -0.1;

    //particles, device side
    struct particle particlesD;
    memoryHandling::allocateDeviceParticles(particlesD);
    memoryHandling::synchronizeParticles(particlesD,particlesH,memoryHandling::listOfVariables::All,cudaMemcpyHostToDevice);

    //create the output folder
    std::string output_folder = "output";
    if (std::filesystem::exists(output_folder)) 
    {
        std::filesystem::remove_all(output_folder);
    }
    std::filesystem::create_directory(output_folder);

    //create a file to save the energy
    std::ofstream energy(output_folder + "/energy.csv");
    energy << "Kin.\tPot.\tTot.\n";

    //simulation settings
    int GridSize = (NumberOfParticles + 1)/BlockSize;
    std::cout << "<<<" << GridSize << "," << BlockSize << ">>>\n";
    int numberOfLaunches = (timestep.numberOfSteps+1)/timestep.saveSteps;

    //SIMULATION
    auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < numberOfLaunches; i++)
    {        
        //save energy
        float K = forceHandling::calculateTotalKineticEnergy(particlesH,numberOfActiveParticles);
        float K1 = forceHandling::calculateTranslationEnergy(particlesH,numberOfActiveParticles);
        float K2 = forceHandling::calculateRotationalEnergy(particlesH,numberOfActiveParticles);
        float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,numberOfActiveParticles);
        energy << K1 << "\t" <<  K2 << "\t" << P << "\t" << K+P << "\n";


        //print info
        /*if(i%10==0)
        {
            std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
            std::cout << "K="<< K << "\t P=" << P << "\t T=" << K+P << "\n";
        }*/


        //save
        std::string name = output_folder + "/test_" + std::to_string(i) + ".vtu";
        ioHandling::saveParticlesVTK(numberOfActiveParticles,particlesH,name);

        //solve
        void *kernelArgs[] = {
            (void*)&particlesD,
            (void*)&numberOfActiveParticles,
            (void*)&materials,
            (void*)&timestep,
            (void*)&gravity,
            (void*)&BCs,
            (void*)&i
        };
        cudaLaunchCooperativeKernel((void*)solver, GridSize, BlockSize, kernelArgs);
        CHECK(cudaDeviceSynchronize());

        //copy D2H
        memoryHandling::synchronizeParticles(
            particlesH,
            particlesD,
            memoryHandling::listOfVariables::Position,
            cudaMemcpyDeviceToHost
        );
        memoryHandling::synchronizeParticles(
            particlesH,
            particlesD,
            memoryHandling::listOfVariables::Velocity,
            cudaMemcpyDeviceToHost
        );
        memoryHandling::synchronizeParticles(
            particlesH,
            particlesD,
            memoryHandling::listOfVariables::AngularVelocity,
            cudaMemcpyDeviceToHost
        );
    }
    //save
    std::string name = output_folder + "/test_" + std::to_string(numberOfLaunches) + ".vtu";
    ioHandling::saveParticlesVTK(numberOfActiveParticles,particlesH,name);

    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "Runtime: " << duration/1000 << " ms" << std::endl;

    energy.flush();
    energy.close();

    memoryHandling::freeHostParticles(particlesH);
    memoryHandling::freeDeviceParticles(particlesD);

}