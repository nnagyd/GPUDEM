/**
 * @file ex2_layered_deposition.cu
 * @author Dániel NAGY
 * @version 1.0
 * @brief Gravitational deposition in layers example
 * @date 2023.09.12.
 * 
 * This code simulates the deposition of N particles.
 * The particles are deposited in layers. Material data
 *  - R = 40mm ± 10mm
 *  - E = 2G = 200GPa
 *  - Rho = 2000 kg/m^3
 *  - mu =  0.5, mu0 = 0.7, mur = 0.02
 *  - e  = 0.1
 * Domain
 *  - Layout = 2m x 2m
*/


#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>

constexpr int NumberOfParticles = 8192;
constexpr int NumberOfMaterials = 1;
constexpr int NumberOfBoundaries = 5;

#include "source/solver.cuh"

int particlesPerLayer = 1024;
int numberOfLayers = 8 + 1;


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

    //set the initial particle distribution
    struct particleDistribution pdist;
    pdist.min.x = -1.0f;
    pdist.max.x =  1.0f;
    pdist.min.y = -1.0f;
    pdist.max.y =  1.0f;
    pdist.min.z = 2.2f;
    pdist.max.z = 3.6f;
    pdist.vmean = 0.0f;
    pdist.vsigma= 0.00f;
    pdist.Rmean = 0.04f;
    pdist.Rsigma= 0.01f;

    //material parameters
    struct materialParameters materials;
    materials.rho[0] = 2000.0f;
    materials.E[0]   = 2.0e8f;
    materials.G[0]   = 1.0e8f;
    materials.nu[0]  = 0.3f;
    materials.e[0]   = 0.1f;
    materials.mu[0]  = 0.5f;
    materials.mu0[0] = 0.7f;
    materials.mur[0] = 0.02f;

    materialHandling::calculateMaterialContact(materials,materialHandling::methods::Min,materialHandling::methods::HarmonicMean,materialHandling::methods::HarmonicMean);
    materialHandling::printMaterialInfo(materials,true);

    //timestep settings
    float dt = 2e-4f;
    int saves = 500;
    struct timestepping timestep(0.0f,5.0f,dt,saves);

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
        BCs.beta[i] = 0.0f;
    }

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    particleHandling::generateParticleLocation(
        particlesH,
        pdist,
        particleHandling::ParticleSizeDistribution::Uniform,
        particleHandling::ParticleVelocityDistribution::Uniform);
    particleHandling::generateParticleParameters(particlesH,materials,0,0,NumberOfParticles);

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
                (void*)&materials,
                (void*)&timestep,
                (void*)&gravity,
                (void*)&BCs,
                (void*)&i
            };
            int GridSize = (numberOfActiveParticles + 1)/BlockSize;
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