/**
 * @file ex1_deposition.cu
 * @author Dániel NAGY
 * @version 1.0
 * @brief Gravitational deposition example
 * @date 2023.09.12.
 * 
 * This code simulates the deposition of 8192 particles. The wall uses the Hertz
 * model too.
 * The particles are stored in the data/ex3_input.vtu input file.
 *  - E = G = 20MPa
 *  - Rho = 1000 kg/m^3
 *  - mu =  0.5, mu0 = 0.7, mur = 0.02
 *  - beta = 1.5
 * Domain
 *  - Layout = 2m x 2m
*/


#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>

constexpr int NumberOfParticles = 8192;
constexpr int NumberOfMaterials = 2;
constexpr int NumberOfBoundaries = 5;


#include "source/solver.cuh"


int main(int argc, char const *argv[])
{
    //Set GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    //cudaFuncSetAttribute(solver, cudaFuncAttributePreferredSharedMemoryCarveout, 0);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    //set the initial particle distribution
    struct particleDistribution pdist;
    pdist.min.x = -1.0f;
    pdist.max.x =  1.0f;
    pdist.min.y = -1.0f;
    pdist.max.y =  1.0f;
    pdist.min.z = 0.0f;
    pdist.max.z = 4.0f;
    pdist.vmean = 0.00f;
    pdist.vsigma= 0.00f;
    pdist.Rmean = 0.03f;
    pdist.Rsigma= 0.005f;

    //material parameters
    struct materialParameters materials;
    //particles
    materials.rho[0]=1000.0f;
    materials.E[0] = 2.0e5f;
    materials.G[0] = 1.0e5f;
    materials.nu[0] = 0.3f;
    materials.e[0] = 0.1f;
    materials.mu[0] = 0.6f;
    materials.mu0[0] = 0.7f;
    materials.mur[0] = 0.05f;
    //walls
    materials.E[1] = 2.0e8f;
    materials.G[1] = 1.0e8f;
    materials.nu[1] = 0.3f;
    materials.e[1] = 0.1f;
    materials.mu[1] = 0.8f;
    materials.mu0[1] = 0.9f;
    materials.mur[1] = 0.2f;

    materialHandling::calculateMaterialContact(materials,materialHandling::methods::Min,materialHandling::methods::HarmonicMean,materialHandling::methods::HarmonicMean);
    //materialHandling::printMaterialInfo(materials,true);


    //timestep settings
    float dt = 1.0e-4f;
    float saves = 0.05f;
    struct timestepping timestep(0.0f,2.5001f,dt,saves);

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
        BCs.material[i] = 1;
    }

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    particleHandling::generateParticleLocation(
        particlesH,
        pdist,
        particleHandling::ParticleSizeDistribution::Uniform,
        particleHandling::ParticleVelocityDistribution::Uniform);
    //ioHandling::readParticlesVTK(particlesH,"data/ex3_input.vtu",NumberOfParticles);
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
    int GridSize = (NumberOfParticles + 1)/BlockSize;
    std::cout << "<<<" << GridSize << "," << BlockSize << ">>>\n";
    int numberOfLaunches = (timestep.numberOfSteps+1)/timestep.saveSteps;

    //SIMULATION
    auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < numberOfLaunches; i++)
    {

        //save
        std::string name = output_folder + "/test_" + std::to_string(i) + ".vtu";
        ioHandling::saveParticlesVTK(NumberOfParticles,particlesH,name);

        //solve
        /*void *kernelArgs[] = {
            (void*)&particlesD,
            (void*)&NumberOfParticles,
            (void*)&materials,
            (void*)&timestep,
            (void*)&gravity,
            (void*)&BCs,
            (void*)&i
        };
        cudaLaunchCooperativeKernel((void*)solver, GridSize, BlockSize, kernelArgs);*/
        solver<<<GridSize,BlockSize>>>(particlesD,NumberOfParticles,materials,timestep,gravity,BCs,i);
        CHECK(cudaDeviceSynchronize());

        //print info
        if(i%1==0)
        {
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

            //save energy
            float K = forceHandling::calculateTotalKineticEnergy(particlesH,NumberOfParticles);
            float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,NumberOfParticles);

            std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
            std::cout << "K = " << K << "\t P = " << P << "\t T =" << K+P << "\n";

            energy << K << "\t" << P << "\t" << K+P << "\n";

            
        }
    }
    auto endTime = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    std::cout << "Runtime: " << duration/1000 << " ms" << std::endl;

    energy.flush();
    energy.close();

    memoryHandling::freeHostParticles(particlesH);
    memoryHandling::freeDeviceParticles(particlesD);

}