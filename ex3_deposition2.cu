/**
 * @file ex1_deposition.cu
 * @author Dániel NAGY
 * @version 1.0
 * @brief Gravitational deposition example
 * @date 2023.08.04.
 * 
 * This code simulates the deposition of 8192 particles. 
 * The particles are stored in the particle8192_INIT.vtu input file
 *  - E = G = 20MPa
 *  - Rho = 1000 kg/m^3
 *  - mu =  0.5, mu0 = 0.7
 *  - beta = 1.5
 * Domain
 *  - Layout = 2m x 2m
*/


#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>

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
    struct materialParameters pars;
    pars.rho=1000.0f;
    pars.E = 20000.0f;
    pars.G = 20000.0f;
    pars.nu = 0.3f;
    pars.beta = 10.0f;
    pars.mu = 0.5f;
    pars.mu0 = 0.7f;

    //timestep settings
    float dt = 2.5e-4f;
    float saves = 0.05f;
    struct timestepping timestep(0.0f,50.0f,dt,saves);

    //body forces
    struct bodyForce gravity;
    gravity.x = 0.0f;
    gravity.y = 0.0f;
    gravity.z = -9.81f;

    //BCs - wall on the bottom at z=0 and around in a 2m x 2m domain
    struct boundaryCondition BCs;
    BCs.n[0] = vec3D(0.0f,0.0f,-1.0f); BCs.p[0] = vec3D(0.0f,0.0f,0.0f);  
    BCs.n[1] = vec3D(-1.0f,0.0f,0.0f); BCs.p[1] = vec3D(-1.0f,0.0f,0.0f);
    BCs.n[2] = vec3D( 1.0f,0.0f,0.0f); BCs.p[2] = vec3D( 1.0f,0.0f,0.0f);
    BCs.n[3] = vec3D(0.0f,-1.0f,0.0f); BCs.p[3] = vec3D(0.0f,-1.0f,0.0f);
    BCs.n[4] = vec3D(0.0f, 1.0f,0.0f); BCs.p[4] = vec3D(0.0f, 1.0f,0.0f);
    for(int i = 0; i < NumberOfBoundaries; i++)
    {
        BCs.type[i] = BoundaryConditionType::ReflectiveWall; 
        BCs.alpha[i] =  0.8f; 
        BCs.beta[i] = 0.003f;
    }

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    ioHandling::readParticlesVTK(particlesH,"data/ex3_input.vtu");
    particleHandling::generateParticleParameters(particlesH,pars);

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
        //print info
        if(i%1==0)
        {
            std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
        }
        
        //save energy
        float K = forceHandling::calculateTotalKineticEnergy(particlesH,NumberOfParticles);
        float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,NumberOfParticles);
        energy << K << "\t" << P << "\t" << K+P << "\n";
        std::cout << K << "\t" << P << "\t" << K+P << "\n";

        //save
        std::string name = output_folder + "/test_" + std::to_string(i) + ".vtu";
        ioHandling::saveParticlesVTK(NumberOfParticles,particlesH,name);

        //solve
        void *kernelArgs[] = {
            (void*)&particlesD,
            (void*)&NumberOfParticles,
            (void*)&pars,
            (void*)&timestep,
            (void*)&gravity,
            (void*)&BCs,
            (void*)&i
        };
        cudaLaunchCooperativeKernel((void*)solver, GridSize, BlockSize, kernelArgs);

        //solver<<<GridSize,BlockSize>>>(particlesD,NumberOfParticles,pars,timestep,gravity,BCs,i);
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
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "Runtime: " << duration/1000 << " ms" << std::endl;

    energy.flush();
    energy.close();

    memoryHandling::freeHostParticles(particlesH);
    memoryHandling::freeDeviceParticles(particlesD);

}