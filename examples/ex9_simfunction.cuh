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



float SimulationFunction(int id, float rho, float E, float nu, float e, float mu, float mur, float depth)
{
    int numberOfActiveParticles = NumberOfParticles;
    
    //std::cout << "id="<<id <<"\t rho="<<rho<<"\t E="<< E << "\t nu=" << nu << "\t e=" << e <<"\t mu=" << mu <<"\n";
    //body forces
    struct bodyForce gravity;
    gravity.x = 0.0f;
    gravity.y = 0.0f;
    gravity.z = -9.81f;

    //BCs - wall on the bottom at z=0 and around in a 2m x 2m domain
    struct boundaryCondition BCsH;
    struct boundaryCondition BCsD;
    struct particle particlesH;
    struct particle particlesD;
    struct materialParameters materials;

    ioHandling::readGeometrySTL(BCsH,0,BoundaryConditionType::HertzWall,1,1.0f,"data/ex9_walls.stl");
    ioHandling::readGeometrySTL(BCsH,sizeWalls,BoundaryConditionType::HertzWall,1,1.0f,"data/ex9_tool.stl");
    domainHandling::translateBoundaryConditions(BCsH,sizeWalls,sizeWalls+sizeTool,-0.52f,0.0f,0.375f-depth,true);

    domainHandling::convertBoundaryConditions(BCsH,BCsD);
    memoryHandling::allocateDeviceBoundary(BCsH,BCsD);

    memoryHandling::allocateHostParticles(particlesH);
    memoryHandling::allocateDeviceParticles(particlesD);

    /*
    ---------------------------- Anyagbeállítás --------------------------------
    */

    //material parameters
    materials.rho[0]= rho;
    materials.E[0] = E;
    materials.G[0] = 0.5f*E/(1.0f+nu);
    materials.nu[0] = nu;
    materials.e[0] = e;
    materials.mu[0] = mu;
    materials.mu0[0] = mu*1.1f;
    materials.mur[0] = mur;

    //tool parameters
    materials.rho[1]= 4000.0f;
    materials.E[1] = 2.0e8f;
    materials.G[1] = 0.76923e8f;
    materials.nu[1] = 0.3f;
    materials.e[1] = e;
    materials.mu[1] = mu;
    materials.mu0[1] = mu*1.1f;
    materials.mur[1] = mur;

    materialHandling::calculateMaterialContact(materials,materialHandling::methods::Min,materialHandling::methods::HarmonicMean,materialHandling::methods::HarmonicMean);

    /*
    ---------------------------- Ülepítés --------------------------------
    */

    //timestep settings
    float dt = 0.5e-4f;
    int savesteps = 400;
    struct timestepping timestep(0.0f,2.01f,dt,savesteps);


    //particles, host side
    ioHandling::readParticlesVTK(particlesH,"data/ex9_input_2.vtu",numberOfActiveParticles);
    particleHandling::generateParticleParameters(particlesH,materials,0,0,numberOfActiveParticles);

    //particles, device side
    memoryHandling::synchronizeParticles(particlesD,particlesH,memoryHandling::listOfVariables::All,cudaMemcpyHostToDevice);

    //SIMULATION
    int numberOfLaunches = (timestep.numberOfSteps+1)/timestep.saveSteps;
    int GridSize = (numberOfActiveParticles + 1)/BlockSize;
    int counter = 0;
    bool depositionReady = false;
    auto startTime = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < numberOfLaunches; i++)
    {
        //solve 
        void *kernelArgs[] = {
            (void*)&particlesD,
            (void*)&numberOfActiveParticles,
            (void*)&materials,
            (void*)&timestep,
            (void*)&gravity,
            (void*)&BCsD,
            (void*)&i
        };
        cudaLaunchCooperativeKernel((void*)solver, GridSize, BlockSize, kernelArgs);
        CHECK(cudaDeviceSynchronize());
        
        /*std::string name = output_folder + "/test_" + std::to_string(i) + ".vtu";
        ioHandling::saveParticlesVTK(numberOfActiveParticles,particlesH,name);*/

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

        //save energy
        float K = forceHandling::calculateTotalKineticEnergy(particlesH,numberOfActiveParticles);
        float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,numberOfActiveParticles);

        std::cout << "K = " << K << "\n";
        
        if(K < 3.0f)
        {
            counter++;
            if(counter == 3)
            {
                //std::cout << " --> Deposition ready in "<< i << " iteratons! K = " << K << "\n";
                depositionReady = true;
                break;
            }
        }
        else
        {
            counter = 0;
        }
    }
    if(!depositionReady)
    {
        std::cout << "ERROR IN THE DEPOSITION!\n";
    }


    /*
    ---------------------------- Tető levágása --------------------------------
    */

    for(int i = 0; i < NumberOfParticles - 1; i++)
    {
        for(int j = 0; j < NumberOfParticles - 1 - i; j++)
        {
            if(particlesH.u.z[j] > particlesH.u.z[j+1])
            {
                float x,y,z,vx,vy,vz,R;
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
    }

    for(int i = 0; i < NumberOfParticles; i++)
    {
        if(particlesH.u.z[i] > 0.3f)
        {
            numberOfActiveParticles = i;
            break;
        } 
    }


    //std::cout << " --> Particle reordered. Np = " << numberOfActiveParticles << "\n";    

    /*
    ---------------------------- Mozgatás --------------------------------
    */
    //timestep settings
    dt = 0.5e-4f;
    savesteps = 10;
    timestep = timestepping(0.0f,1.80f,dt,savesteps);

    //energia és erő
    std::ofstream energy("output/energy_"+ std::to_string(id) +".csv");
    std::ofstream force("output/force_"+ std::to_string(id) +".csv");

    //particles, host side
    particleHandling::generateParticleParameters(particlesH,materials,0,0,numberOfActiveParticles);

    //particles, device side
    memoryHandling::synchronizeParticles(particlesD,particlesH,memoryHandling::listOfVariables::All,cudaMemcpyHostToDevice);

    //SIMULATION
    numberOfLaunches = (timestep.numberOfSteps+1)/timestep.saveSteps;
    GridSize = (numberOfActiveParticles + 1)/BlockSize;
    vec3D Fkum(0.0f,0.0f,0.0f);
    int countFsaves = 0;
    for(int i = 0; i < numberOfLaunches; i++)
    {

        //solve 
        void *kernelArgs[] = {
            (void*)&particlesD,
            (void*)&numberOfActiveParticles,
            (void*)&materials,
            (void*)&timestep,
            (void*)&gravity,
            (void*)&BCsD,
            (void*)&i
        };
        cudaLaunchCooperativeKernel((void*)solver, GridSize, BlockSize, kernelArgs);
        CHECK(cudaDeviceSynchronize());
        
        //move the geometry
        domainHandling::translateBoundaryConditions(BCsH,sizeWalls,sizeWalls+sizeTool,0.00035f,0.0f,0.0f);
        domainHandling::translateBoundaryConditions(BCsD,sizeWalls,sizeWalls+sizeTool,0.00035f,0.0f,0.0f);

        //save 
        if(i%20==0)
        {
            std::string name = "output/test_" + std::to_string(i) + ".vtu";
            std::string name2 = "output/test_" + std::to_string(i) + ".stl";
            ioHandling::saveParticlesVTK(numberOfActiveParticles,particlesH,name);
            ioHandling::writeGeometrySTL(BCsH,sizeWalls,sizeWalls+sizeTool,name2);

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
            memoryHandling::synchronizeBoundary(BCsH,BCsD);

            //save energy
            float K = forceHandling::calculateTotalKineticEnergy(particlesH,numberOfActiveParticles);
            float P = forceHandling::calculateTotalPotentialEnergy(particlesH,gravity,numberOfActiveParticles);
            
            //save force
            vec3D Fsum = vec3D(0.0f,0.0f,0.0f);
            for(int j = sizeWalls; j < NumberOfBoundaries; j++)
            {
                Fsum = Fsum + BCsH.F[j];
                BCsH.F[j].x = 0.0f;
                BCsH.F[j].y = 0.0f;
                BCsH.F[j].z = 0.0f;
            }
            Fsum = Fsum*0.01f; //* (1.0f / (timestep.saveSteps*20));
            force << Fsum.x << "," << Fsum.y << "," << Fsum.z << "\n";
            std::cout << Fsum.x << "," << Fsum.y << "," << Fsum.z << "\n";
            energy << K << "," << P << "," << K+P << "\n";

            force.flush();
            energy.flush();

            memoryHandling::synchronizeBoundary(BCsD,BCsH,cudaMemcpyHostToDevice);

            if(i >= 800)
            {
                Fkum = Fkum + Fsum;
                countFsaves++;
            }
        }
    }
    //calculate F avg
    float Favg = Fkum.x / countFsaves;

    //std::cout << " --> Average force Fx = " << Favg << " from "<< countFsaves <<" steps \n";    


    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    //std::cout << "Runtime: " << duration/1000 << " ms" << std::endl;

    memoryHandling::freeHostParticles(particlesH);
    memoryHandling::freeDeviceParticles(particlesD);

    return Favg;

}