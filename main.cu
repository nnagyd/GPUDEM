/**
 * @file main.cu
 * @author Dániel NAGY
 * @version 1.0
 * @brief Main part, case definition
 * @date 2023.07.20.
 * 
 * Handling of I/O, calling functions...
*/


#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <chrono>

#include "source/solver.cuh"


int main(int argc, char const *argv[])
{

    // get device information
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	cudaSetDevice(dev);

    //begin
    std::cout << "This is GPUDEM 1.0" << std::endl;
    //RandomGeneration::initializeRandomSeed();

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
    pdist.Rmean = 0.03f;
    pdist.Rsigma = 0.01f;

    //timestep settings
    struct timestepping timestep(0.0f,5.0f,0.00002f,2000);

    //material parameters
    struct materialParameters pars;
    pars.rho[0]=1000.0f;
    pars.E[0] = 20000.0f;
    pars.G[0] = 20000.0f;
    pars.nu[0] = 0.3f;
    pars.e[0] = 0.6f;
    pars.mu[0] = 0.5f;
    pars.mu0[0] = 0.5f;
    materialHandling::calculateMaterialContact(pars,materialHandling::methods::Min,materialHandling::methods::HarmonicMean,materialHandling::methods::HarmonicMean);
    materialHandling::printMaterialInfo(pars,true);

    //body forces
    struct bodyForce gravity;
    gravity.x = 0.0f;
    gravity.y = 0.0f;
    gravity.z = -9.81f;

    //BCs - wall on the bottom at z=0 and around in a 2x2 domain
    struct boundaryCondition BCs;
    BCs.n[0] = vec3D(0.0f,0.0f,-1.0f); BCs.p[0] = vec3D(0.0f,0.0f,0.0f);  BCs.type[0] = BoundaryConditionType::ReflectiveWall; BCs.alpha[0] =  0.9f; BCs.beta[0] = 0.08f;
    BCs.n[1] = vec3D(-1.0f,0.0f,0.0f); BCs.p[1] = vec3D(-1.0f,0.0f,0.0f); BCs.type[1] = BoundaryConditionType::ReflectiveWall; BCs.alpha[1] =  0.6f; BCs.beta[1] = 0.08f;
    BCs.n[2] = vec3D( 1.0f,0.0f,0.0f); BCs.p[2] = vec3D( 1.0f,0.0f,0.0f); BCs.type[2] = BoundaryConditionType::ReflectiveWall; BCs.alpha[2] =  0.6f; BCs.beta[2] = 0.08f;
    BCs.n[3] = vec3D(0.0f,-1.0f,0.0f); BCs.p[3] = vec3D(0.0f,-1.0f,0.0f); BCs.type[3] = BoundaryConditionType::ReflectiveWall; BCs.alpha[3] =  0.6f; BCs.beta[3] = 0.08f;
    BCs.n[4] = vec3D(0.0f, 1.0f,0.0f); BCs.p[4] = vec3D(0.0f, 1.0f,0.0f); BCs.type[4] = BoundaryConditionType::ReflectiveWall; BCs.alpha[4] =  0.6f; BCs.beta[4] = 0.08f;

    //particles, host side
    struct particle particlesH;
    memoryHandling::allocateHostParticles(particlesH);
    particleHandling::generateParticleLocation(particlesH,pdist);
    //ioHandling::readParticlesVTK(particlesH,"particle8192_INIT.vtu");
    particleHandling::generateParticleParameters(particlesH,pars,0,0,NumberOfParticles-1);
    
    //particles, device side
    struct particle particlesD;
    memoryHandling::allocateDeviceParticles(particlesD);

    //generate and synchronize
    memoryHandling::synchronizeParticles(particlesD,particlesH,memoryHandling::listOfVariables::All,cudaMemcpyHostToDevice);

    //create the output folder
    std::string output_folder = "output";
    if (std::filesystem::exists(output_folder)) 
    {
        std::filesystem::remove_all(output_folder);
    }
    std::filesystem::create_directory(output_folder);

    //simulation settings
    int GridSize = (NumberOfParticles + 1)/BlockSize;
    std::cout << "<<<" << GridSize << "," << BlockSize << ">>>\n";
    int numberOfLaunches = (timestep.numberOfSteps+1)/timestep.saveSteps;
    int numberOfActiveParticles = NumberOfParticles;

    //SIMULATION
    auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < numberOfLaunches; i++)
    {
        //print info
        std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
        float K = forceHandling::calculateTotalKineticEnergy(particlesH,numberOfActiveParticles);
        float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,numberOfActiveParticles);
        std::cout << "K = " << K << "\tP = " << P << "\tT = " << K+P << "\n";

        //save
        std::string name = output_folder + "/test_" + std::to_string(i) + ".vtu";
        ioHandling::saveParticlesVTK(numberOfActiveParticles,particlesH,name);

        //solve
        solver<<<GridSize,BlockSize>>>(particlesD,numberOfActiveParticles,pars,timestep,gravity,BCs,i);
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
    std::cout << "Runtime: " << duration/1000/1000 << " seconds" << std::endl;

    memoryHandling::freeHostParticles(particlesH);
    memoryHandling::freeDeviceParticles(particlesD);
}