/**
 * @file ex2_layered_deposition.cu
 * @author Dániel NAGY
 * @version 1.0
 * @brief Gravitational deposition in layers example
 * @date 2023.08.04.
 * 
 * This code simulates the deposition of N particles.
 * The particles are deposited in layers. Material data
 *  - R = 30mm ± 10mm
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
    int numBlocksPerSm = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, solver, BlockSize, 0);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    printf("#max.blocks/SM=%d #SM = %d \n", numBlocksPerSm,deviceProp.multiProcessorCount);
    printf("#max.particles = %d\n",numBlocksPerSm*deviceProp.multiProcessorCount*BlockSize);
    cudaSetDevice(dev);

    //particles per layer
    int particlesPerLayer = 4096;
    int numberOfLayers = 12 + 4;

    //set the initial particle distribution
    struct particleDistribution pdist;
    pdist.min.x = -1.0f;
    pdist.max.x =  1.0f;
    pdist.min.y = -1.0f;
    pdist.max.y =  1.0f;
    pdist.min.z = 2.2f;
    pdist.max.z = 3.6f;
    pdist.vmean = 0.0f;
    pdist.vsigma = 0.00f;
    pdist.Rmean = 0.02f;
    pdist.Rsigma = 0.005f;

    //material parameters
    struct materialParameters pars;
    pars.rho=1000.0f;
    pars.E = 20000.0f;
    pars.G = 20000.0f;
    pars.nu = 0.3f;
    pars.beta = 20.0f;
    pars.mu = 0.5f;
    pars.mu0 = 0.7f;

    //timestep settings
    float dt = 1e-4f;
    float saves = 0.05f;
    struct timestepping timestep(0.0f,1.0f,dt,saves);

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
        BCs.alpha[i] =  0.6f; 
        BCs.beta[i] = 0.04f;
    }

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    particleHandling::generateParticleLocation(particlesH,pdist);
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
    int numberOfLaunches = (timestep.numberOfSteps+1)/timestep.saveSteps;

    int numberOfActiveParticles = particlesPerLayer;

    //SIMULATION
    auto startTime = std::chrono::high_resolution_clock::now();
    for(int j = 0; j < numberOfLayers; j++)
    {
        std::cout << "-------- Layer " << j << "/" << numberOfLayers <<  " p=" << numberOfActiveParticles << " --------\n";
        for(int i = 0; i < numberOfLaunches; i++)
        {        
            //save energy
            float K = forceHandling::calculateTotalKineticEnergy(particlesH,numberOfActiveParticles);
            float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,numberOfActiveParticles);
            energy << K << "\t" << P << "\t" << K+P << "\n";

            //print info
            if(i%1==0)
            {
                std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\t K=" << K << "\t P=" << P << "\t T=" << K+P << "\n";
            }

            //save
            std::string name = output_folder + "/test_" + std::to_string(i + j*numberOfLaunches) + ".vtu";
            ioHandling::saveParticlesVTK(numberOfActiveParticles,particlesH,name);

            //solver
            void *kernelArgs[] = {
                (void*)&particlesD,
                (void*)&numberOfActiveParticles,
                (void*)&pars,
                (void*)&timestep,
                (void*)&gravity,
                (void*)&BCs,
                (void*)&i
            };
            int GridSize = (numberOfActiveParticles + 1)/BlockSize;
            //solver<<<GridSize,BlockSize>>>(particlesD,numberOfActiveParticles,pars,timestep,gravity,BCs,i);
            CHECK(cudaLaunchCooperativeKernel((void*)solver, GridSize, BlockSize, kernelArgs));
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
        }//end of launch

        //increase the number of active particles and cap
        numberOfActiveParticles += particlesPerLayer;
        if(numberOfActiveParticles > NumberOfParticles)
        {
            numberOfActiveParticles = NumberOfParticles;
        }
    }//end of all layers
    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "Runtime: " << duration/1000 << " ms" << std::endl;

    energy.flush();
    energy.close();

    memoryHandling::freeHostParticles(particlesH);
    memoryHandling::freeDeviceParticles(particlesD);

}