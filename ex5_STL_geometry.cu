/**
 * @file ex1_deposition.cu
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

constexpr int NumberOfParticles = 2048;
constexpr int NumberOfMaterials = 2;
constexpr int NumberOfBoundaries = 16;

#include "source/solver.cuh"

#define SQ2 0.7071067812f

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
    materials.E[1] = 200000.0f;
    materials.G[1] = 200000.0f;
    materials.nu[1] = 0.3f;
    materials.e[1] = 0.1f;
    materials.mu[1] = 0.6f;
    materials.mu0[1] = 0.7f;
    materials.mur[1] = 0.03f;

    materialHandling::calculateMaterialContact(materials,materialHandling::methods::Min,materialHandling::methods::HarmonicMean,materialHandling::methods::HarmonicMean);
    materialHandling::printMaterialInfo(materials,true);

    //particle distribution
    struct particleDistribution pdist;
    pdist.min.x = -1.0f;
    pdist.max.x =  1.0f;
    pdist.min.y = -1.0f;
    pdist.max.y =  1.0f;
    pdist.min.z = 1.5f;
    pdist.max.z = 5.0f;
    pdist.vmean = 0.0f;
    pdist.vsigma = 0.00f;
    pdist.Rmean = 0.04f;
    pdist.Rsigma = 0.02f;

    //timestep settings
    float dt = 5e-5f;
    float saves = 0.05f;
    struct timestepping timestep(0.0f,10.0f,dt,saves);

    //body forces
    struct bodyForce gravity;
    gravity.x = 0.0f;
    gravity.y = 0.0f;
    gravity.z = -9.81f;

    //BCs - wall on the bottom at z=0 and around in a 2m x 2m domain
    struct boundaryCondition BCs;
    //bottom wall
    BCs.n[0] = vec3D(0.0f,0.0f,-1.0f); BCs.p[0] = vec3D(1.0f,-1.0f,0.0f); 
    BCs.s[0] = vec3D(-2.0f,0.0f,0.0f); BCs.t[0] = vec3D(0.0f,2.0f,0.0f); 

    BCs.n[1] = vec3D(0.0f,0.0f,-1.0f); BCs.p[1] = vec3D(-1.0f,1.0f,0.0f);
    BCs.s[1] = vec3D(2.0f,0.0f,0.0f); BCs.t[1] = vec3D(0.0f,-2.0f,0.0f); 

    //sides
    BCs.n[2] = vec3D(1.0f,0.0f,0.0f); BCs.p[2] = vec3D(1.0f,-1.0f,0.0f);
    BCs.s[2] = vec3D(0.0f,0.0f,8.0f);  BCs.t[2] = vec3D(0.0f,2.0f,0.0f);

    BCs.n[3] = vec3D(1.0f,0.0f,0.0f); BCs.p[3] = vec3D(1.0f,1.0f,0.0f);
    BCs.s[3] = vec3D(0.0f,0.0f,8.0f);  BCs.t[3] = vec3D(0.0f,-2.0f,8.0f);

    BCs.n[4] = vec3D(0.0f,1.0f,0.0f); BCs.p[4] = vec3D(1.0f,1.0f,0.0f);
    BCs.s[4] = vec3D(0.0f,0.0f,8.0f); BCs.t[4] = vec3D(-2.0f,0.0f,0.0f);

    BCs.n[5] = vec3D(0.0f,1.0f,0.0f); BCs.p[5] = vec3D(-1.0f,1.0f,0.0f);
    BCs.s[5] = vec3D(0.0f,0.0f,8.0f); BCs.t[5] = vec3D(2.0f,0.0f,8.0f);

    BCs.n[6] = vec3D(-1.0f,0.0f,0.0f); BCs.p[6] = vec3D(-1.0f,1.0f,0.0f);
    BCs.s[6] = vec3D(0.0f,0.0f,8.0f); BCs.t[6] = vec3D(0.0f,-2.0f,0.0f);

    BCs.n[7] = vec3D(-1.0f,0.0f,0.0f); BCs.p[7] = vec3D(-1.0f,-1.0f,0.0f);
    BCs.s[7] = vec3D(0.0f,0.0f,8.0f); BCs.t[7] = vec3D(0.0f,2.0f,8.0f);

    BCs.n[8] = vec3D(0.0f,-1.0f,0.0f); BCs.p[8] = vec3D(-1.0f,-1.0f,0.0f);
    BCs.s[8] = vec3D(0.0f,0.0f,8.0f);  BCs.t[8] = vec3D(2.0f,0.0f,0.0f);

    BCs.n[9] = vec3D(0.0f,-1.0f,0.0f); BCs.p[9] = vec3D(1.0f,-1.0f,0.0f);
    BCs.s[9] = vec3D(0.0f,0.0f,8.0f);  BCs.t[9] = vec3D(-2.0f,0.0f,8.0f);

    //V shape sides 1
    BCs.n[10] = vec3D(0.0f,-1.0f,0.0f); BCs.p[10] = vec3D(0.0f,-0.5f,0.6f);
    BCs.s[10] = vec3D(-0.4f,0.0f,0.4f); BCs.t[10] = vec3D(0.4f,0.0f,0.4f);

    BCs.n[11] = vec3D(0.0f,1.0f,0.0f); BCs.p[11] = vec3D(0.0f,0.5f,0.6f);
    BCs.s[11] = vec3D(-0.4f,0.0f,0.4f); BCs.t[11] = vec3D(0.4f,0.0f,0.4f);

    //V shape sides 2
    BCs.n[12] = vec3D(-SQ2,0.0f,-SQ2); BCs.p[12] = vec3D(0.0f,0.5f,0.6f);
    BCs.s[12] = vec3D(0.0f,-1.0f,0.0f); BCs.t[12] = vec3D(-0.4f,0.0f,0.4f);

    BCs.n[13] = vec3D(SQ2,0.0f,-SQ2); BCs.p[13] = vec3D(0.0f,0.5f,0.6f);
    BCs.s[13] = vec3D(0.0f,-1.0f,0.0f); BCs.t[13] = vec3D(0.4f,0.0f,0.4f);

    BCs.n[14] = vec3D(-SQ2,0.0f,-SQ2);  BCs.p[14] = vec3D(-0.4f,-0.5f,1.0f);
    BCs.s[14] = vec3D(0.0f,1.0f,0.0f); BCs.t[14] = vec3D(0.4f,0.0f,-0.4f);

    BCs.n[15] = vec3D(SQ2,0.0f,-SQ2);  BCs.p[15] = vec3D(0.4f,-0.5f,1.0f);
    BCs.s[15] = vec3D(0.0f,1.0f,0.0f); BCs.t[15] = vec3D(-0.4f,0.0f,-0.4f);

    for(int i = 0; i < NumberOfBoundaries; i++)
    {
        BCs.type[i] = BoundaryConditionType::HertzWall; 
        BCs.material[i] =  1;
        float sn = BCs.s[i].length();
        BCs.s_scale[i] = 1.0f / (sn*sn);
        float tn = BCs.t[i].length();
        BCs.t_scale[i] = 1.0f / (tn*tn);
        std::cout << "i = " << i << "\t s_scale = " << BCs.s_scale[i] << "\t t_scale = " << BCs.t_scale[i] << "\n";
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
        if(i%1==0)
        {
            std::cout << "Launch " << i << "\t/ " << numberOfLaunches << "\n";
            std::cout << "K="<< K << "\t P=" << P << "\t T=" << K+P << "\n";
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