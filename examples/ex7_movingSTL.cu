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

constexpr int NumberOfParticles = 16384;
constexpr int NumberOfMaterials = 3;

constexpr int sizeMoving = 10;
constexpr int sizeWalls = 10;
constexpr int NumberOfBoundaries = sizeMoving + sizeWalls;

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
    materials.rho[0]= 300.0f;
    materials.E[0] = 10000.0f;
    materials.G[0] = 10000.0f;
    materials.nu[0] = 0.3f;
    materials.e[0] = 0.001f;
    materials.mu[0] = 0.6f;
    materials.mu0[0] = 0.7f;
    materials.mur[0] = 0.03f;

    //walls
    materials.rho[1]= 1000.0f;
    materials.E[1] = 200000.0f;
    materials.G[1] = 200000.0f;
    materials.nu[1] = 0.3f;
    materials.e[1] = 0.1f;
    materials.mu[1] = 0.6f;
    materials.mu0[1] = 0.7f;
    materials.mur[1] = 0.03f;

    //stl
    materials.rho[2]= 4000.0f;
    materials.E[2] = 2000000.0f;
    materials.G[2] = 2000000.0f;
    materials.nu[2] = 0.3f;
    materials.e[2] = 0.2f;
    materials.mu[2] = 0.6f;
    materials.mu0[2] = 0.7f;
    materials.mur[2] = 0.03f;

    materialHandling::calculateMaterialContact(materials,materialHandling::methods::Min,materialHandling::methods::HarmonicMean,materialHandling::methods::HarmonicMean);
    materialHandling::printMaterialInfo(materials,true);

    //timestep settings
    float dt = 5e-5f;
    float saves = 0.0005f;
    struct timestepping timestep(0.0f,10.0f,dt,saves);

    //body forces
    struct bodyForce gravity;
    gravity.x = 0.0f;
    gravity.y = 0.0f;
    gravity.z = -9.81f;

    //BCsH - wall on the bottom at z=0 and around in a 2m x 2m domain
    struct boundaryCondition BCsH;
    struct boundaryCondition BCsD;

    ioHandling::readGeometrySTL(BCsH,0,BoundaryConditionType::HertzWall,1,1.0f,"data/ex7_walls.stl");
    ioHandling::readGeometrySTL(BCsH,sizeWalls,BoundaryConditionType::HertzWall,2,1.0f,"data/ex7_moving.stl");

    domainHandling::printBoundaryConditions(BCsH);
    domainHandling::convertBoundaryConditions(BCsH,BCsD);
    domainHandling::translateBoundaryConditions(BCsH,sizeWalls,sizeWalls+sizeMoving,0.0f,0.0f,0.0f);

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    ioHandling::readParticlesVTK(particlesH,"data/ex7_input.vtu");
    particleHandling::generateParticleParameters(particlesH,materials,0,0,NumberOfParticles);

    //particles, device side
    struct particle particlesD;
    memoryHandling::allocateDeviceParticles(particlesD);
    memoryHandling::synchronizeParticles(particlesD,particlesH,memoryHandling::listOfVariables::All,cudaMemcpyHostToDevice);

    //boundary, device and host side of forces
    memoryHandling::allocateDeviceBoundary(BCsD);
    BCsH.F = new vec3D[NumberOfBoundaries];

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
        //solve
        void *kernelArgs[] = {
            (void*)&particlesD,
            (void*)&NumberOfParticles,
            (void*)&materials,
            (void*)&timestep,
            (void*)&gravity,
            (void*)&BCsH,
            (void*)&i
        };
        cudaLaunchCooperativeKernel((void*)solver, GridSize, BlockSize, kernelArgs);
        CHECK(cudaDeviceSynchronize());

        if(i % 50 == 0)
        {      
            //save
            std::string name = output_folder + "/test_" + std::to_string(i) + ".vtu";
            std::string name2 = output_folder + "/test_" + std::to_string(i) + ".stl";
            ioHandling::saveParticlesVTK(NumberOfParticles,particlesH,name);
            ioHandling::writeGeometrySTL(BCsH,sizeWalls,sizeWalls+sizeMoving,name2);

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
            memoryHandling::synchronizeBoundary(BCsH.F,BCsD);

            //save energy
            float K = forceHandling::calculateTotalKineticEnergy(particlesH,NumberOfParticles);
            float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,NumberOfParticles);
            energy << K << "\t" << P << "\t" << K+P << "\n";

            std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
            std::cout << "K="<< K << "\t P=" << P << "\t T=" << K+P << "\n";

            //save force
            vec3D Fsum = vec3D(0.0f,0.0f,0.0f);
            for(int j = sizeMoving; j < NumberOfBoundaries; j++)
            {
                Fsum = Fsum + BCsH.F[j];
                //std::cout << "  -> F[" << j <<"] = (" << F[j].x << "," << F[j].y << "," << F[j].z << ")\n";
            }
            std::cout << "F = (" << Fsum.x << "," << Fsum.y << "," << Fsum.z << ")\n";
        }
        
        if(true)
        {
            domainHandling::translateBoundaryConditions(BCsH,sizeWalls,sizeWalls+sizeMoving,-0.0005f,0.0f,0.0f);
            domainHandling::translateBoundaryConditions(BCsD,sizeWalls,sizeWalls+sizeMoving,-0.0005f,0.0f,0.0f);
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