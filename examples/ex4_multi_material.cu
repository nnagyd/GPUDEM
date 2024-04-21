/**
 * @file ex1_deposition.cu
 * @author Dániel NAGY
 * @version 1.0
 * @brief Gravitational deposition example
 * @date 2023.09.12.
 * 
 * This code simulates the deposition of a denser and a lighter material.
 * The dense material has rho=1000kg/m3 and light material has rho=200kg/m3
 *
 * Domain
 *  - Layout = 2m x 2m
*/


#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>

constexpr int NumberOfParticles = 4096;
constexpr int NumberOfMaterials = 3;
constexpr int NumberOfBoundaries = 5;

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
    //material 0
    materials.rho[0]= 200.0f;
    materials.E[0] = 10000.0f;
    materials.G[0] = 10000.0f;
    materials.nu[0] = 0.3f;
    materials.e[0] = 0.001f;
    materials.mu[0] = 0.6f;
    materials.mu0[0] = 0.7f;
    materials.mur[2] = 0.02f;

    //material 2
    materials.rho[1]=1000.0f;
    materials.E[1] = 20000.0f;
    materials.G[1] = 20000.0f;
    materials.nu[1] = 0.3f;
    materials.e[1] = 0.002f;
    materials.mu[1] = 0.4f;
    materials.mu0[1] = 0.5f;
    materials.mur[2] = 0.02f;

    //wall
    materials.E[2] = 200000.0f;
    materials.G[2] = 200000.0f;
    materials.nu[2] = 0.3f;
    materials.e[2] = 0.002f;
    materials.mu[2] = 0.7f;
    materials.mu0[2] = 0.8f;
    materials.mur[2] = 0.05f;

    materialHandling::calculateMaterialContact(materials,materialHandling::methods::Min,materialHandling::methods::HarmonicMean,materialHandling::methods::HarmonicMean);
    materialHandling::printMaterialInfo(materials,true);

    //particle distribution
    struct particleDistribution pdist;
    pdist.min.x = -1.0f;
    pdist.max.x =  1.0f;
    pdist.min.y = -1.0f;
    pdist.max.y =  1.0f;
    pdist.min.z = 0.5f;
    pdist.max.z = 5.0f;
    pdist.vmean = 0.0f;
    pdist.vsigma = 0.00f;
    pdist.Rmean = 0.05f;
    pdist.Rsigma = 0.01f;

    //timestep settings
    float dt = 1e-4f;
    float saves = 0.05f;
    struct timestepping timestep(0.0f,15.0f,dt,saves);

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
        BCs.type[i] = BoundaryConditionType::HertzWall; 
        BCs.material[i] = 2;
    }

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    particleHandling::generateParticleLocation(
        particlesH,
        pdist,
        particleHandling::ParticleSizeDistribution::Uniform,
        particleHandling::ParticleVelocityDistribution::Uniform);
    particleHandling::generateParticleParameters(particlesH,materials,0,0,NumberOfParticles/2);
    particleHandling::generateParticleParameters(particlesH,materials,1,NumberOfParticles/2,NumberOfParticles);

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
        float K = forceHandling::calculateTotalKineticEnergy(particlesH,NumberOfParticles);
        float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,NumberOfParticles);
        energy << K << "\t" << P << "\t" << K+P << "\n";


        //print info
        if(i%10==0)
        {
            std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
            std::cout << "K="<< K << "\t P=" << P << "\t T=" << K+P << "\n";
        }

        if(i == 100)
        {
            gravity.x = 3.0f;
            gravity.z = 12.0f;
        }
        if(i == 105)
        {
            gravity.x = -3.0f;
            gravity.z = -12.0f;
        }
        if(i == 110)
        {
            gravity.x = 0.0f;
            gravity.z = -9.81f;
        }
        if(i == 150)
        {
            gravity.x = 3.0f;
            gravity.z = 12.0f;
        }
        if(i == 155)
        {
            gravity.x = -3.0f;
            gravity.z = -12.0f;
        }
        if(i == 160)
        {
            gravity.x = 0.0f;
            gravity.z = -9.81f;
        }

        //save
        std::string name = output_folder + "/test_" + std::to_string(i) + ".vtu";
        ioHandling::saveParticlesVTK(NumberOfParticles,particlesH,name);

        //solve
        void *kernelArgs[] = {
            (void*)&particlesD,
            (void*)&NumberOfParticles,
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
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "Runtime: " << duration/1000 << " ms" << std::endl;

    energy.flush();
    energy.close();

    memoryHandling::freeHostParticles(particlesH);
    memoryHandling::freeDeviceParticles(particlesD);

}