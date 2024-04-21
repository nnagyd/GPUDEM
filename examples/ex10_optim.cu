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

constexpr int NumberOfParticles = 65536;
constexpr int NumberOfMaterials = 2;

constexpr int sizeWalls = 10;
constexpr int sizeTool = 500;
constexpr int NumberOfBoundaries = sizeWalls + sizeTool;

#include "source/solver.cuh"
#include "ex10_simfunction.cuh"


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
    std::string depositions = "output/deposition/";
    if (std::filesystem::exists(output_folder)) 
    {
        std::filesystem::remove_all(output_folder);
    }
    std::filesystem::create_directory(output_folder);
    std::filesystem::create_directory(depositions);

    /*
    Fixed parameters
    */
    float depth = 0.15f;
    float targetF = 310.0f;
    float v=0.7f;

    /*float rho = 1993;
    float E = 8.96e6;
    float nu = 0.227;
    float e = 0.104;
    float mu = 0.684;
    float mur = 0.282;*/

    float rho = 2379;
    float E = 2.305e6;
    float nu = 0.475;
    float e = 0.258;
    float mu = 0.637;
    float mur = 0.277;

    int np = SimulationFunctionDeposition(0, rho, E, nu, e, mu, mur, depth);
    SimulationFunctionCultivator(0, rho, E, nu, e, mu, mur, v, depth, np);




}