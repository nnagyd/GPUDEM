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

constexpr int NumberOfParticles = 8576;
constexpr int NumberOfMaterials = 1;
constexpr int NumberOfBoundaries = 5;

constexpr int numberOfActiveParticles = 8519;

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
    materials.mur[0] = 0.01f;
    materials.mu[0] = 0.5f;
    materials.mu0[0] = 0.01f;

    materialHandling::calculateMaterialContact(materials,
        materialHandling::methods::Min, //friction
        materialHandling::methods::HarmonicMean, //E, G
        materialHandling::methods::Min); //damping
    materials.pairing[0].G_star[0] = materials.G[0];


    //timestep settings
    float dt = 1e-4f;
    int saves = 100;
    struct timestepping timestep(0.0f,1.0f,dt,saves);

    //body forces
    struct bodyForce gravity;
    gravity.x = 0.0f;
    gravity.y = 0.0f;
    gravity.z = -9.81f;

    //BCs - wall on the bottom at z=0 and around in a 2m x 2m domain
    struct boundaryCondition BCs;
    BCs.n[0] = vec3D(0.0f,0.0f,-1.0f); BCs.p[0] = vec3D(0.0f,0.0f,-1.5f);  
    BCs.n[1] = vec3D(-1.0f,0.0f,0.0f); BCs.p[1] = vec3D(-1.5f,0.0f,0.0f);
    BCs.n[2] = vec3D( 1.0f,0.0f,0.0f); BCs.p[2] = vec3D( 1.5f,0.0f,0.0f);
    BCs.n[3] = vec3D(0.0f,-1.0f,0.0f); BCs.p[3] = vec3D(0.0f,-0.5f,0.0f);
    BCs.n[4] = vec3D(0.0f, 1.0f,0.0f); BCs.p[4] = vec3D(0.0f, 0.5f,0.0f);
    for(int i = 0; i < NumberOfBoundaries; i++)
    {
        BCs.type[i] = BoundaryConditionType::HertzWall; 
        BCs.material[i] = 0;
    }

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    ioHandling::readParticlesCSV(particlesH,"data/ex6_40.dat",numberOfActiveParticles);
    particleHandling::generateParticleParameters(particlesH,materials,0,0,numberOfActiveParticles);

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
        float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,numberOfActiveParticles);
        energy << K << "\t" << P << "\t" << K+P << "\n";


        //print info
        if(i%10==0)
        {
            std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
            std::cout << "K="<< K << "\t P=" << P << "\t T=" << K+P << "\n";
        }


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