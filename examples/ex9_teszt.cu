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

constexpr int NumberOfParticles = 38912;
constexpr int NumberOfMaterials = 2;

constexpr int sizeWalls = 10;
constexpr int sizeTool = 500;
constexpr int NumberOfBoundaries = sizeWalls + sizeTool;

#include "source/solver.cuh"
#include "ex9_simfunction.cuh"


int main(int argc, char const *argv[])
{
    //Set GPU
    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    cudaSetDevice(dev);
    RandomGeneration::initializeRandomSeed();


    //create the output folder
    std::string output_folder = "output";
    if (std::filesystem::exists(output_folder)) 
    {
        std::filesystem::remove_all(output_folder);
    }
    std::filesystem::create_directory(output_folder);

    /*
    Fixed parameters
    */
    float depth = 0.15f;
    float targetF = 310.0f;
   
    float rho = 2400;
    float E = 2.979e6;
    float nu = 0.45;
    float e = 0.063;
    float mu = 0.684;
    float mur = 0.229;
    /*
    float rho = 2379;
    float E = 2.305e6;
    float nu = 0.475;
    float e = 0.258;
    float mu = 0.637;
    float mur = 0.277;
    */
    /*float rho = 1900;
    float E = 5.00e6;
    float nu = 0.1;
    float e = 0.1;
    float mu = 0.5;
    float mur = 0.25;*/
    
    /*float rho = 2387;
    float E = 7.978e6;
    float nu = 0.486;
    float e = 0.113;
    float mu = 0.495;
    float mur = 0.264;*/
    

    float force = SimulationFunction(0, rho, E, nu, e, mu, mur, depth);
    std::cout << "Force = "<<force << "\n";




}