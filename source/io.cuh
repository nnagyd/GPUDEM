/**
 * @file io.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Input-output handling
 * @date 2023.09.12.
 * 
 * Functions to do it
*/

#ifndef io_H
#define io_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>
#include "settings.cuh"
#include "material.cuh"
#include "math.cuh"

/**
 * \brief Contains all the functions for writing and reading data
 */
namespace ioHandling
{
    /**
    * \brief Save the data of a list of particles in a .particle textfile
    *
    * @param numberOfActiveParticles Number of active parameters
    * @param particles List of particles
    * @param location File location
    */
    void saveParticles(int numberOfActiveParticles, struct particle particles, std::string location)
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
        for(int i = 0; i < numberOfActiveParticles; i++)
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
    * @param numberOfActiveParticles Number of active parameters
    * @param particles List of particles
    * @param location File location
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

    /**
    * \brief Reads the particle data from a vtk compatible .vtu unstructured grid file.
    *
    * @param particles List of particles
    * @param location File location
    *
    * @return Number of particles in the file
    */
    int readParticlesVTK(struct particle particles, std::string location, int numberOfActiveParticles)
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

                        std::istringstream iss(line);
                        iss >> particles.R[idx];
                        idx++;
                        if(idx>=numberOfActiveParticles) break;
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

    /**
    * \brief Reads the particle data from a .csv file.
    *
    * @param particles List of particles
    * @param location File location
    *
    * @return Number of particles in the file
    */
    int readParticlesCSV(struct particle particles, std::string location, int numberOfActiveParticles)
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
            if(idx>=numberOfActiveParticles) break;
        }

        file.close();

        return idx;
    }//end of read particles

    /**
    * \brief Contains data from STL
    */
    struct Triangle {
        float n[3];
        float v1[3];
        float v2[3];
        float v3[3];
        uint16_t attribute_byte_count;
    };

    /**
    * \brief Reads an STL geometry and adds it to the BC struct
    *
    * @param BC List of boundary condition
    * @param startId Index where the given STL file starts in the BC struct
    * @param BCtype Type of the boundary condition defined by the STL
    * @param materialId Material corresponding to the given STL
    * @param scale Scaling the model up/down
    * @param location File location
    *
    * @return Number of triangles in the file
    */
    int readGeometrySTL(struct boundaryCondition &BC, int startId, BoundaryConditionType BCtype, int materialId, var_type scale, std::string location)
    {
        std::ifstream file(location, std::ios::in);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open the file" << std::endl;
            return 1;
        }
    
        //list of triangle
        std::vector<Triangle> triangles;
        std::string line;
    
        //skip the first line containing header information
        for (int i = 0; i < 1; ++i) {
            std::getline(file, line);
        }
    
        //go through the file
        while (std::getline(file, line)) 
        {
            //if file ended break
            if(line.find("endsolid") != std::string::npos)
            {
                //end of file
                break;
            }

            //initialize a triangle
            Triangle triangle;
            sscanf(line.c_str(), "facet normal %f %f %f", &triangle.n[0], &triangle.n[1], &triangle.n[2]);

            //skip line with text "outer loop"
            std::getline(file, line);
            
            //read vertices
            for (int i = 0; i < 3; ++i) {
                std::getline(file, line);
                sscanf(line.c_str(), "vertex %f %f %f", &triangle.v1[i], &triangle.v2[i], &triangle.v3[i]);
            }
            
            //skip line with text "endloop" and "endfacet"
            std::getline(file, line);
            std::getline(file, line);
    
            //add the triangle to the vector
            triangles.push_back(triangle);
        }
    
        file.close();
    
        // Process the triangle data
        for (int i = 0; i < triangles.size(); i++) {

            if(i + startId == NumberOfBoundaries) 
            {
                break;
            }
            /*printf("i=%d \tn=(%6.3lf,%6.3lf,%6.3lf)\tp=(%6.3lf,%6.3lf,%6.3lf)\n",
                i,
                triangles[i].n[0],triangles[i].n[1],triangles[i].n[2],
                triangles[i].v1[0],triangles[i].v2[0],triangles[i].v3[0]);*/

            //read the normal vector
            BC.n[i+startId] = vec3D( triangles[i].n[0],triangles[i].n[1],triangles[i].n[2]);

            //read the p vector
            BC.p[i+startId] = vec3D( triangles[i].v1[0],triangles[i].v2[0],triangles[i].v3[0]);

            //calculate s and t
            BC.s[i+startId] = vec3D(    triangles[i].v1[1] - triangles[i].v1[0],
                                        triangles[i].v2[1] - triangles[i].v2[0],
                                        triangles[i].v3[1] - triangles[i].v3[0]);
            BC.t[i+startId] = vec3D(    triangles[i].v1[2] - triangles[i].v1[0],
                                        triangles[i].v2[2] - triangles[i].v2[0],
                                        triangles[i].v3[2] - triangles[i].v3[0]);

            //apply scaling
            BC.p[i+startId] = BC.p[i+startId]*scale;
            BC.s[i+startId] = BC.s[i+startId]*scale;
            BC.t[i+startId] = BC.t[i+startId]*scale;

            //set type and material
            BC.material[i+startId] = materialId;
            BC.type[i+startId] = BCtype;

            /*printf("i=%d \tn=(%6.3lf,%6.3lf,%6.3lf)\tp=(%6.3lf,%6.3lf,%6.3lf)\n",
                i,
                BC.n[i+startId].x,BC.n[i+startId].y,BC.n[i+startId].z,
                BC.p[i+startId].x,BC.p[i+startId].y,BC.p[i+startId].z);*/
        }

        return triangles.size();
    }


        /**
    * \brief Reads an STL geometry and adds it to the BC struct
    *
    * @param BC List of boundary condition
    * @param startId Index where the given STL file starts in the BC struct
    * @param BCtype Type of the boundary condition defined by the STL
    * @param materialId Material corresponding to the given STL
    * @param scale Scaling the model up/down
    * @param location File location
    *
    * @return Number of triangles in the file
    */
    void writeGeometrySTL(struct boundaryCondition &BC, int startId, int endId, std::string location)
    {
        std::ofstream file(location);

        if (!file.is_open()) {
            std::cerr << "Error: Could not open the file" << std::endl;
            return;
        }
    
        file << "solid DEMgeoFile" << std::endl;
    
        for(int i = startId; i < endId; i++)
        {
            file << "facet normal " << BC.n[i].x << " " << BC.n[i].y << " " << BC.n[i].z << "\n"; 
            file << "outer loop\n"; 
            file << "vertex " << BC.p[i].x << " " << BC.p[i].y << " " << BC.p[i].z << "\n"; 
            file << "vertex " << BC.p[i].x + BC.t[i].x << " " << BC.p[i].y + BC.t[i].y << " " << BC.p[i].z + BC.t[i].z  << "\n"; 
            file << "vertex " << BC.p[i].x + BC.s[i].x << " " << BC.p[i].y + BC.s[i].y << " " << BC.p[i].z + BC.s[i].z  << "\n";
            file << "endloop\n"; 
            file << "endfacet\n";  
        }
        
        file << "endsolid DEMgeoFile" << std::endl;

        file.flush();
        file.close();

        return;
    }
}

#endif