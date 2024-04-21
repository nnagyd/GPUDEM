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

float fitness(float F1, float F2, float F3, float f1, float f2, float f3)
{
    return sqrt((F1-f1)*(F1-f1)  +  (F2-f2)*(F2-f2)  +   (F3-f3)*(F3-f3));
}

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
    float v1 = 0.706f;
    float F1 = 309.5f;
    float v2 = 1.222f;
    float F2 = 358.1f;
    float v3 = 1.854f;
    float F3 = 400.1f;

    /*
    Define stuff for DE
    */
    float CR = 0.5f; //crossover rate
    float F = 0.8f; //differential weight
    const int Npopulation = 10;
    const int Nparameters = 6;
    const int Niters = 500;
    float mins[Nparameters] = {500.0f,   4e4f,0.001f,0.05f,0.05f,0.01f};
    float maxs[Nparameters] = {2500.0f,1.0e7f,0.5f,0.95f,0.7f,0.30f};
    bool expgen[Nparameters] = {false,true,false,false,false,false};
    float currentPos[Npopulation][Nparameters];
    float currentVal[Npopulation][3];
    float currentFitness[Npopulation];
    float bestPos[Nparameters];
    float bestFitness = 1e9;
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
        int np = SimulationFunctionDeposition(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], depth);
        currentVal[i][0] = SimulationFunctionCultivator(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], v1, depth, np);
        currentVal[i][1] = SimulationFunctionCultivator(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], v2, depth, np);
        currentVal[i][2] = SimulationFunctionCultivator(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], v3, depth, np);
        currentFitness[i] = fitness(F1,F2,F3,currentVal[i][0],currentVal[i][1],currentVal[i][2]);
        evals++;

        //report value
        report << evals << "\t";
        for(int j = 0; j < Nparameters; j++)
        {
            report << currentPos[i][j] << "\t";
        }
        report << currentVal[i][0] << "\t" << currentVal[i][1]  << "\t" << currentVal[i][2] << "\n";
        report.flush();

        //new best?
        if(currentFitness[i] <= bestFitness)
        {
            std::cout << " --> new best!\n";
            bestFitness = currentFitness[i];
            best << evals << "\t";
            for(int j = 0; j < Nparameters; j++)
            {
                bestPos[j] = currentPos[i][j];
                best << currentPos[i][j] << "\t";
            }
            best << bestFitness << "\t" << currentVal[i][0] << "\t" << currentVal[i][1]  << "\t" << currentVal[i][2] << "\n";
            best.flush();
        }
    }

    std::ofstream generation("results/gen_M.csv");
    for(int i = 0; i < Npopulation; i++)
    {
        generation << i << "\t";
        for(int j = 0; j < Nparameters; j++)
        {
            generation << currentPos[i][j] << "\t";
        }
        generation << currentVal[i][0] << "\t" << currentVal[i][1]  << "\t"<< currentVal[i][2] << "\t" << currentFitness[i] << "\n";
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
            Fsum += currentVal[i][1];
        }
        //create report
        std::cout << "------------- Generation " << iter << "/" << Niters << " ----------------\n";
        std::cout << "  Current best: F = " << bestFitness << "\n";
        std::cout << " Current avg2: Fa = " << Fsum/float(Npopulation) << "\n";
        std::cout << "  --> rho = " << bestPos[0] << "\n";
        std::cout << "  -->   E = " << bestPos[1] << "\n";
        std::cout << "  -->  nu = " << bestPos[2] << "\n";
        std::cout << "  -->   e = " << bestPos[3] << "\n";
        std::cout << "  -->  mu = " << bestPos[4] << "\n";
        std::cout << "  --> mur = " << bestPos[5] << "\n";

        status << iter << "\t" << bestFitness << "\t" <<  Fsum/float(Npopulation) << "\n";
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
            int np = SimulationFunctionDeposition(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], depth);
            float f1 = SimulationFunctionCultivator(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], v1, depth, np);
            float f2 = SimulationFunctionCultivator(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], v2, depth, np);
            float f3 = SimulationFunctionCultivator(evals, currentPos[i][0], currentPos[i][1], currentPos[i][2], currentPos[i][3], currentPos[i][4], currentPos[i][5], v3, depth, np);
            float newFitness = fitness(F1,F2,F3,f1,f2,f3);
            evals++;

            //report value
            report << evals << "\t";
            for(int j = 0; j < Nparameters; j++)
            {
                report << newPos[j] << "\t";
            }
            report << f1 << "\t" << f2 << "\t" << f3 << "\t" << "\n";
            report.flush();

            //improvement?
            if(newFitness <= currentFitness[i])
            {
                std::cout << " --> improved! fitness = " << newFitness <<"\n";
                currentFitness[i] = newFitness;
                currentVal[i][0] = f1;
                currentVal[i][1] = f2;
                currentVal[i][2] = f3;
                for(int j = 0; j < Nparameters; j++)
                {
                    currentPos[i][j] = newPos[j];
                }
            }

            //new best?
            if(newFitness <= bestFitness)
            {
                std::cout << " --> new best!\n";
                bestFitness = newFitness;
                best << evals << "\t";
                for(int j = 0; j < Nparameters; j++)
                {
                    bestPos[j] = newPos[j];
                    best << newPos[j] << "\t";
                }
                best << bestFitness << "\t" << f1 << "\t" << f2  << "\t" << f3 << "\n";
                best.flush();
            }
        }


        std::ofstream generation0("results/gen_"+std::to_string(iter)+".csv");
        for(int i = 0; i < Npopulation; i++)
        {
            generation0 << i << "\t";
            for(int j = 0; j < Nparameters; j++)
            {
                generation0 << currentPos[i][j] << "\t";
            }
            generation0 << currentVal[i][0] << "\t" << currentVal[i][1]  << "\t"<< currentVal[i][2] << "\t" << currentFitness[i] << "\n";
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
        std::cout << "  Runtime: " << duration/1000/1000 << " s" << std::endl;
    }



    best.close();
    status.close();
    report.close();

}