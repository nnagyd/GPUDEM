/**
 * @file io.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Input-output handling
 * @date 2023.08.03.
 * 
 * Functions to do it
*/

#ifndef io_H
#define io_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include "settings.cuh"
#include "material.cuh"
#include "math.cuh"

namespace ioHandling
{

    /**
    * \brief Save the data of a list of particles in a .particle textfile
    *
    * @param particles A list of particles
    */
    void saveParticles(struct particle particles, std::string location)
    {
        std::ofstream out(location);
        out.precision(7);
        out.width(12); out << "x" << ","; 
        out.width(12); out << "y" << ","; 
        out.width(12); out << "z" << ","; 
        out.width(12); out << "v_x" << ","; 
        out.width(12); out << "v_y" << ","; 
        out.width(12); out << "v_z" << ","; 
        out.width(12); out << "size" << "\n"; 
        for(int i = 0; i < NumberOfParticles; i++)
        {
            out.width(12); out << particles.u.x[i] << ","; 
            out.width(12); out << particles.u.y[i] << ","; 
            out.width(12); out << particles.u.z[i] << ","; 
            out.width(12); out << particles.v.x[i] << ","; 
            out.width(12); out << particles.v.y[i] << ","; 
            out.width(12); out << particles.v.z[i] << ","; 
            out.width(12); out << particles.R[i] << "\n"; 
        }
        out.flush();
        out.close();
    }


    /**
    * \brief Save the data of a list of particles as a vtk compatible .vtu unstructured grid file.
    *
    * @param particles A list of particles
    *
    * File can be opened in paraview and the particles can be displayed using the Glyph filter
    */
    void saveParticlesVTK(int numberOfActiveParticles, struct particle particles, std::string location)
    {
        std::ofstream out;
        if(outputFormat == OutputFormat::ASCII)
        {
            out.open(location);
            out.precision(7);
        }
        if(outputFormat == OutputFormat::Binary)
        {
            out.open(location, std::ios::out | std::ios::binary);
        }
        
        //print headers
        out << "<?xml version=\"1.0\"?>\n";
        out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
        out << "\t<UnstructuredGrid>\n";
        out << "\t\t<Piece NumberOfPoints=\""<< numberOfActiveParticles <<"\" NumberOfCells=\"0\">\n";
        out << "\t\t\t<!-- Particle positions -->\n";
        out << "\t\t\t<Points>\n";



        //print points
        if(outputFormat == OutputFormat::ASCII)
        {
            out << "\t\t\t\t<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out << "\t\t\t\t\t";
                out.width(12); out << particles.u.x[i] << " "; 
                out.width(12); out << particles.u.y[i] << " "; 
                out.width(12); out << particles.u.z[i] << "\n"; 
            }
        }
        if(outputFormat == OutputFormat::Binary)
        {
            out << "\t\t\t\t<DataArray type=\"Float64\" NumberOfComponents=\"3\" format=\"binary\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out.write(reinterpret_cast<const char*>(&particles.u.x[i]),sizeof(var_type));
                out.write(reinterpret_cast<const char*>(&particles.u.y[i]),sizeof(var_type));
                out.write(reinterpret_cast<const char*>(&particles.u.z[i]),sizeof(var_type));
            }
        }
        out << "\n\t\t\t\t</DataArray>\n";



        //print middle stuff
        out << "\t\t\t</Points>\n";
        out << "\t\t\t<Cells>\n";
        if(outputFormat == OutputFormat::ASCII)
        {
            out << "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\n";
            out << "\t\t\t\t</DataArray>\n";
            out << "\t\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\n";
            out << "\t\t\t\t</DataArray>\n";
            out << "\t\t\t\t<DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">\n";
            out << "\t\t\t\t</DataArray>\n";
        }
        if(outputFormat == OutputFormat::Binary)
        {
            out << "\t\t\t\t<DataArray type=\"Int32\" Name=\"connectivity\" format=\"binary\">\n";
            out << "\t\t\t\t</DataArray>\n";
            out << "\t\t\t\t<DataArray type=\"Int32\" Name=\"offsets\" format=\"binary\">\n";
            out << "\t\t\t\t</DataArray>\n";
            out << "\t\t\t\t<DataArray type=\"Int32\" Name=\"types\" format=\"binary\">\n";
            out << "\t\t\t\t</DataArray>\n";
        }
        out << "\t\t\t</Cells>\n";
        out << "\t\t\t<!-- Particle data -->\n";
        out << "\t\t\t<PointData Scalars=\"Radius\" Vectors=\"Velocity\">\n";



        //print radii
        out << "\t\t\t\t<!-- Particle radii -->\n";
        if(outputFormat == OutputFormat::ASCII)
        {
            out << "\t\t\t\t<DataArray type=\"Float32\" Name=\"Radius\" format=\"ascii\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out << "\t\t\t\t\t";
                out.width(12); out << particles.R[i] << "\n"; 
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }
        if(outputFormat == OutputFormat::Binary)
        {
            out << "\t\t\t\t<DataArray type=\"Float64\" Name=\"Radius\" format=\"binary\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out.write(reinterpret_cast<const char*>(&particles.R[i]),sizeof(var_type));
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }



        //print velocity
        out << "\t\t\t\t<!-- Particle velocity -->\n";
        if(outputFormat == OutputFormat::ASCII && SaveVelocity)
        {
            out << "\t\t\t\t<DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out << "\t\t\t\t\t";
                out.width(12); out << particles.v.x[i] << " "; 
                out.width(12); out << particles.v.y[i] << " "; 
                out.width(12); out << particles.v.z[i] << "\n"; 
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }
        if(outputFormat == OutputFormat::Binary && SaveVelocity)
        {
            out << "\t\t\t\t<DataArray type=\"Float64\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"binary\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out.write(reinterpret_cast<const char*>(&particles.v.x[i]),sizeof(var_type));
                out.write(reinterpret_cast<const char*>(&particles.v.y[i]),sizeof(var_type));
                out.write(reinterpret_cast<const char*>(&particles.v.z[i]),sizeof(var_type));
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }



        //print angular velocity
        out << "\t\t\t\t<!-- Particle omega -->\n";
        if(outputFormat == OutputFormat::ASCII && SaveAngularVelocity)
        {
            out << "\t\t\t\t<DataArray type=\"Float32\" Name=\"AngularVelocity\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out << "\t\t\t\t\t";
                out.width(12); out << particles.omega.x[i] << " "; 
                out.width(12); out << particles.omega.y[i] << " "; 
                out.width(12); out << particles.omega.z[i] << "\n"; 
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }


        //print force
        out << "\t\t\t\t<!-- Particle forces -->\n";
        if(outputFormat == OutputFormat::ASCII && SaveForce)
        {
            out << "\t\t\t\t<DataArray type=\"Float32\" Name=\"Force\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out << "\t\t\t\t\t";
                out.width(12); out << particles.F.x[i] << " "; 
                out.width(12); out << particles.F.y[i] << " "; 
                out.width(12); out << particles.F.z[i] << "\n"; 
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }


        //print moment
        out << "\t\t\t\t<!-- Particle torques -->\n";
        if(outputFormat == OutputFormat::ASCII && SaveTorque)
        {
            out << "\t\t\t\t<DataArray type=\"Float32\" Name=\"Torque\" NumberOfComponents=\"3\" format=\"ascii\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out << "\t\t\t\t\t";
                out.width(12); out << particles.M.x[i] << " "; 
                out.width(12); out << particles.M.y[i] << " "; 
                out.width(12); out << particles.M.z[i] << "\n"; 
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }


        //print id
        out << "\t\t\t\t<!-- Particle ids -->\n";
        if(outputFormat == OutputFormat::ASCII && SaveId)
        {
            out << "\t\t\t\t<DataArray type=\"Int32\" Name=\"Id\" format=\"ascii\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out << "\t\t\t\t\t";
                out.width(12); out << i << "\n"; 
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }

        //print material
        out << "\t\t\t\t<!-- Particle materials -->\n";
        if(outputFormat == OutputFormat::ASCII && SaveMaterial)
        {
            out << "\t\t\t\t<DataArray type=\"Int32\" Name=\"Material\" format=\"ascii\">\n";
            for(int i = 0; i < numberOfActiveParticles; i++)
            {
                out << "\t\t\t\t\t";
                out.width(12); out << particles.material[i] << "\n"; 
            }
            out << "\n\t\t\t\t</DataArray>\n";
        }

        //end stuff
        out << "\t\t\t</PointData>\n";
        out << "\t\t</Piece>\n";
        out << "\t</UnstructuredGrid>\n";
        out << "</VTKFile>\n";


        out.flush();
        out.close();
    }


    int readParticlesVTK(struct particle particles, std::string location)
    {
        std::ifstream file(location);

        if (!file.is_open())
        {
            std::cerr << "Error: Unable to open the file." << std::endl;
            return -1;
        }

        std::string line;
        bool inPointData = false;
        bool inPoints = false;
        int numParticles = 0;
        int idx = 0;

        while (std::getline(file, line))
        {
            if (line.find("<Piece") != std::string::npos)
            {
                size_t pos = line.find("NumberOfPoints=\"");
                if (pos != std::string::npos)
                {
                    std::istringstream iss(line.substr(pos + 16));
                    iss >> numParticles;
                }
            }

            if (line.find("<PointData") != std::string::npos)
            {
                inPointData = true;
            }

            if (line.find("</PointData") != std::string::npos)
            {
                inPointData = false;
            }

            if (inPointData && line.find("Radius") != std::string::npos && line.find("<DataArray") != std::string::npos)
            {
                size_t pos = line.find(">");
                if (pos != std::string::npos)
                {
                    idx=0;
                    while (std::getline(file, line))
                    {
                        if (line.find("</DataArray>") != std::string::npos)
                            break;

                        //std::cout << "idx=" << idx << "\t" << line << "\n";
                        std::istringstream iss(line);
                        iss >> particles.R[idx];
                        idx++;
                        if(idx>=NumberOfParticles) break;
                    }
                }
            }

            if (line.find("<Points>") != std::string::npos)
            {
                inPoints = true;
            }

            if (line.find("</Points>") != std::string::npos)
            {
                inPoints = false;
            }

            if (inPoints && line.find("<DataArray") != std::string::npos)
            {
                size_t pos = line.find(">");
                if (pos != std::string::npos)
                {
                    idx=0;
                    while (std::getline(file, line))
                    {
                        if (line.find("</DataArray>") != std::string::npos)
                            break;

                        std::istringstream iss(line);
                        iss >> particles.u.x[idx] >> particles.u.y[idx] >> particles.u.z[idx];
                        idx++;
                        if(idx>=NumberOfParticles) break;
                    }
                }
            }
        }//end of while in file

        file.close();

        return numParticles;
    }//end of read particles


    int readParticlesCSV(struct particle particles, std::string location)
    {
        std::ifstream file(location);

        if (!file.is_open())
        {
            std::cerr << "Error: Unable to open the file." << std::endl;
            return -1;
        }

        std::string line;
        int idx = 0;

        while (std::getline(file, line))
        {
            std::istringstream iss(line);
            iss >> particles.u.x[idx] >> particles.u.y[idx] >> particles.u.z[idx] >> particles.R[idx];
            idx++;
            if(idx>=NumberOfParticles) break;
        }

        file.close();

        return idx;
    }//end of read particles
}

#endif