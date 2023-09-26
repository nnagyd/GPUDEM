/**
 * @file ex7_movingSTL.cu
 * @author Dániel NAGY
 * @version 1.0
 * @brief Gravitational deposition example
 * @date 2023.09.12.
 * 
 * This code simulates the deposition of particles with special STL geometry.
 *
*/


#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>

constexpr int NumberOfParticles = 147456;
constexpr int NumberOfMaterials = 2;

constexpr int sizeWalls = 10;
constexpr int NumberOfBoundaries = sizeWalls;


int NumberOfActiveParticles = 147456;
constexpr int ParticlesPerLayer = 147456;

#include "source/solver.cuh"


int main(int argc, char const *argv[])
{
    //Set GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);

    //set the initial particle distribution
    struct particleDistribution pdist;
    pdist.min.x = -0.5f;
    pdist.max.x =  0.5f;
    pdist.min.y = -0.35f;
    pdist.max.y =  0.35f;
    pdist.min.z = 0.0f;
    pdist.max.z = 9.0f;
    pdist.vmean = 0.00f;
    pdist.vsigma= 0.00f;
    pdist.Rmean = 4.4e-3f;
    pdist.Rsigma= 0.8e-3f;

    //material parameters
    struct materialParameters materials;
    materials.rho[0]= 1850.0f;
    materials.E[0] = 2.0e6f;
    materials.G[0] = 1.0e6f;
    materials.nu[0] = 0.38f;
    materials.e[0] = 0.5f;
    materials.mu[0] = 0.6f;
    materials.mu0[0] = 0.7f;
    materials.mur[0] = 0.03f;

    //tool parameters
    materials.rho[1]= 4000.0f;
    materials.E[1] = 2.0e8f;
    materials.G[1] = 1.0e8f;
    materials.nu[1] = 0.3f;
    materials.e[1] = 0.2f;
    materials.mu[1] = 0.6f;
    materials.mu0[1] = 0.7f;
    materials.mur[1] = 0.03f;

    materialHandling::calculateMaterialContact(materials,materialHandling::methods::Min,materialHandling::methods::HarmonicMean,materialHandling::methods::HarmonicMean);

    //timestep settings
    float dt = 1.0e-4f;
    float saves = 0.01f;
    struct timestepping timestep(0.0f,20.0f,dt,saves);

    //body forces
    struct bodyForce gravity;
    gravity.x = 0.0f;
    gravity.y = 0.0f;
    gravity.z = -9.81f;

    //BCs - wall on the bottom at z=0 and around in a 2m x 2m domain
    struct boundaryCondition BCsH;
    struct boundaryCondition BCsD;

    ioHandling::readGeometrySTL(BCsH,0,BoundaryConditionType::HertzWall,1,1.0f,"data/ex8_walls.stl");
    domainHandling::convertBoundaryConditions(BCsH,BCsD);

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    /*particleHandling::generateParticleLocation(
        particlesH,
        pdist,
        particleHandling::ParticleSizeDistribution::Uniform,
        particleHandling::ParticleVelocityDistribution::Uniform);*/
    ioHandling::readParticlesVTK(particlesH,"data/ex8_input_147k3.vtu",NumberOfParticles);

    std:: cout << "v = " << particlesH.v.x[0] << "\n";
    //sort based on location z
    /*for(int i = 0; i < NumberOfParticles - 1; i++)
    {
        if(i % 25000 == 0) printf("Sorted: %d\n",i);
        for(int j = 0; j < NumberOfActiveParticles - 1 - i; j++)
        {
            if(particlesH.u.z[j] > particlesH.u.z[j+1])
            {
                float x,y,z,R;
                x = particlesH.u.x[j];
                y = particlesH.u.y[j];
                z = particlesH.u.z[j];
                vx = particlesH.u.x[j];
                vy = particlesH.u.y[j];
                vz = particlesH.u.z[j];
                R = particlesH.R[j];

                particlesH.u.x[j] = particlesH.u.x[j+1];
                particlesH.u.y[j] = particlesH.u.y[j+1];
                particlesH.u.z[j] = particlesH.u.z[j+1];
                particlesH.v.x[j] = particlesH.v.x[j+1];
                particlesH.v.y[j] = particlesH.v.y[j+1];
                particlesH.v.z[j] = particlesH.v.z[j+1];
                particlesH.R[j] = particlesH.R[j+1];

                particlesH.u.x[j+1] = x;
                particlesH.u.y[j+1] = y;
                particlesH.u.z[j+1] = z;
                particlesH.v.x[j+1] = vx;
                particlesH.v.y[j+1] = vy;
                particlesH.v.z[j+1] = vz;
                particlesH.R[j+1] = R;
            }
        }
    }*/

    particleHandling::generateParticleParameters(particlesH,materials,0,0,NumberOfParticles);

    //particles, device side
    struct particle particlesD;
    memoryHandling::allocateDeviceParticles(particlesD);
    memoryHandling::synchronizeParticles(particlesD,particlesH,memoryHandling::listOfVariables::All,cudaMemcpyHostToDevice);

    //boundary, device and host side of forces
    memoryHandling::allocateDeviceBoundary(BCsH,BCsD);

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


    //SIMULATION
    auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < numberOfLaunches; i++)
    {
        int GridSize = (NumberOfActiveParticles + 1)/BlockSize;
        //solve
        /*void *kernelArgs[] = {
            (void*)&particlesD,
            (void*)&NumberOfParticles,
            (void*)&materials,
            (void*)&timestep,
            (void*)&gravity,
            (void*)&BCsD,
            (void*)&i
        };
        cudaLaunchCooperativeKernel((void*)solver, GridSize, BlockSize, kernelArgs);
        CHECK(cudaDeviceSynchronize());*/

        solver<<<GridSize,BlockSize>>>(particlesD,NumberOfActiveParticles,materials,timestep,gravity,BCsD,i);
        CHECK(cudaDeviceSynchronize());

        //save
        std::string name = output_folder + "/test_" + std::to_string(i) + ".vtu";
        ioHandling::saveParticlesVTK(NumberOfActiveParticles,particlesH,name);

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
        float K = forceHandling::calculateTotalKineticEnergy(particlesH,NumberOfActiveParticles);
        float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,NumberOfActiveParticles);
        energy << K << "\t" << P << "\t" << K+P << "\n";

        std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
        std::cout << "K="<< K << "\t P=" << P << "\t T=" << K+P << "\n";

        if(i > 0 && i%12 == 0)
        {
            NumberOfActiveParticles += ParticlesPerLayer;
            if(NumberOfActiveParticles > NumberOfParticles) NumberOfActiveParticles = NumberOfParticles;

            std::cout << "!!!! Number of particles = " << NumberOfActiveParticles << "\n\n";
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