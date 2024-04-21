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

    /*
    Define stuff for DE
    */
    float CR = 0.5f; //crossover rate
    float F = 0.8f; //differential weight
    const int Npopulation = 60;
    const int Nparameters = 6;
    const int Niters = 500;
    float mins[Npopulation] = {500.0f,   4e4f,0.001f,0.05f,0.05f,0.01f};
    float maxs[Npopulation] = {2500.0f,1.0e7f,0.5f,0.95f,0.7f,0.30f};
     bool expgen[Npopulation] = {false,true,false,false,false,false};
    float currentPos[Npopulation][Nparameters];
    float currentVal[Npopulation];
    float bestPos[Nparameters];
    float bestVal = 0;
    int evals = 0; //counts the number of F evaluations

    /*
    output file
    */
    std::ofstream status("results/status.csv");
    std::ofstream report("results/report.csv");
    std::ofstream best("results/best.csv");

    /*
    Initialize with random positions
    */
    std::cout << "------------- Initialization ----------------\n";
    for(int i = 0; i < Npopulation; i++)
    {
        std::cout << "agent " << i << "/" << Npopulation << "\n";
        for(int j = 0; j < Nparameters; j++)
        {
            if(!expgen[j])
            {
                currentPos[i][j] = RandomGeneration::randomInRange(mins[j],maxs[j]);
            }
            else
            {
                currentPos[i][j] = exp(RandomGeneration::randomInRange(log(mins[j]),log(maxs[j])));
            }
        }

        //evaluate the function
        currentVal[i] = SimulationFunction(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], depth);
        evals++;

        //report value
        report << evals << "\t";
        for(int j = 0; j < Nparameters; j++)
        {
            report << currentPos[i][j] << "\t";
        }
        report << currentVal[i] << "\n";
        report.flush();

        //new best?
        if(abs(currentVal[i]-targetF) <= abs(bestVal-targetF))
        {
            std::cout << " --> new best!\n";
            bestVal = currentVal[i];
            best << evals << "\t";
            for(int j = 0; j < Nparameters; j++)
            {
                bestPos[j] = currentPos[i][j];
                best << currentPos[i][j] << "\t";
            }
            best << bestVal << "\n";
            best.flush();
        }
    }

    /*
    Iterations
    */
    for(int iter = 0; iter < Niters; iter++)
    {
        //calculate average force
        float Fsum = 0.0f;
        for(int i = 0; i < Npopulation; i++)
        {
            Fsum += currentVal[i];
        }
        //create report
        std::cout << "------------- Generation " << iter << "/" << Niters << " ----------------\n";
        std::cout << "  Current best: F = " << bestVal << "\n";
        std::cout << "  Current avg: Fa = " << Fsum/float(Npopulation) << "\n";
        std::cout << "  --> rho = " << bestPos[0] << "\n";
        std::cout << "  -->   E = " << bestPos[1] << "\n";
        std::cout << "  -->  nu = " << bestPos[2] << "\n";
        std::cout << "  -->   e = " << bestPos[3] << "\n";
        std::cout << "  -->  mu = " << bestPos[4] << "\n";
        std::cout << "  --> mur = " << bestPos[5] << "\n";

        status << iter << "\t" << bestVal << "\t" <<  Fsum/float(Npopulation) << "\n";
        status.flush();


        auto startTime = std::chrono::high_resolution_clock::now();


        //for each agent
        for(int i = 0; i < Npopulation; i++)
        {
            std::cout << "agent " << i << "/" << Npopulation << "\n";

            //pick 3 random agents
            int a_idx = RandomGeneration::randomInRange(0,Npopulation,i);
            int b_idx = RandomGeneration::randomInRange(0,Npopulation,i,a_idx);
            int c_idx = RandomGeneration::randomInRange(0,Npopulation,i,a_idx,b_idx);

            int R = RandomGeneration::randomInRange(0,Nparameters);

            //new positions
            float newPos[Nparameters];
            for(int j = 0; j < Nparameters; j++)
            {
                float r = RandomGeneration::randomInRange(0.0001f,1.0f);
                if(j == R || r < CR) //we have a crossover
                {
                    newPos[j] = currentPos[a_idx][j] + F*(currentPos[b_idx][j]-currentPos[c_idx][j]);

                    //apply restrictions
                    if(newPos[j] > maxs[j])
                    {
                        newPos[j] = maxs[j];
                    }
                    if(newPos[j] < mins[j])
                    {
                        newPos[j] = mins[j];
                    }
                }
                else
                {
                    newPos[j] = currentPos[i][j];
                }
            }

            //new value
            float newVal = SimulationFunction(evals, newPos[0], newPos[1], newPos[2], newPos[3], newPos[4], newPos[5], depth);
            evals++;

            //report value
            report << evals << "\t";
            for(int j = 0; j < Nparameters; j++)
            {
                report << newPos[j] << "\t";
            }
            report << newVal << "\n";
            report.flush();

            //improvement?
            if(abs(newVal-targetF) <= abs(currentVal[i]-targetF))
            {
                std::cout << " --> improved! F = " << newVal <<"\n";
                currentVal[i] = newVal;
                for(int j = 0; j < Nparameters; j++)
                {
                    currentPos[i][j] = newPos[j];
                }
            }

            //new best?
            if(abs(newVal-targetF) <= abs(bestVal-targetF))
            {
                std::cout << " --> new best!\n";
                bestVal = newVal;
                best << evals << "\t";
                for(int j = 0; j < Nparameters; j++)
                {
                    bestPos[j] = newPos[j];
                    best << newPos[j] << "\t";
                }
                best << bestVal << "\n";
                best.flush();
            }
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        std::cout << "  Runtime: " << duration/1000/1000 << " s" << std::endl;
    }



    best.close();
    status.close();
    report.close();

}