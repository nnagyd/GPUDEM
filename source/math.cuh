/**
 * @file math.cuh
 * @author Dániel NAGY
 * @version 1.0
 * @brief Math function
 * @date 2023.07.24.
 * 
 * Contains common functions
*/


#ifndef math_H
#define math_H

#include <cmath>

#define CHECK(call) \
{ \
  const cudaError_t error = call; \
  if (error != cudaSuccess) \
  { \
  printf("Error: %s:%d, ", __FILE__, __LINE__); \
  printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
  exit(-10*error); \
  } \
}


/* 
 * \brief Contains all the constant
 */
namespace constant
{
    //constants for force calculations
    constexpr var_type CONST_PI = 3.141592653589793238462643;
    constexpr var_type VOLUME_FACTOR = CONST_PI * 4.0 / 3.0;
    constexpr var_type DAMPING = -1.8257418583505537115;  //-2.0 * sqrt(5.0/6.0);

    //numbers
    constexpr var_type ZERO = 0.0;
    constexpr var_type NUMBER_04 = 0.4;
    constexpr var_type NUMBER_05 = 0.5;
    constexpr var_type NUMBER_4o3 = 4.0/3.0;
    constexpr var_type NUMBER_1 = 1.0;
    constexpr var_type NUMBER_2 = 2.0;
    constexpr var_type NUMBER_8 = 8.0;

    //constants in numerical methods
    constexpr var_type AB2C1 = 1.5;
    constexpr var_type AB2C2 = 0.5;
    constexpr var_type AM2C1 = 0.5;
}


struct vector{
    var_type x;
    var_type y;
    var_type z;

    __device__ __host__ vector(var_type x=constant::ZERO, var_type y=constant::ZERO, var_type z=constant::ZERO): x(x), y(y), z(z) {};

    //addition
    __device__ vector operator+(const vector& other) const 
    {
        return vector(x + other.x, y + other.y, z + other.z);
    }

    //subtraction
    __device__ vector operator-(const vector& other) const 
    {
        return vector(x - other.x, y - other.y, z - other.z);
    }

    //dot product
    __device__ var_type operator*(const vector& other) const 
    {
        return x * other.x + y * other.y + z * other.z;
    }

    //multiplication with scalar
    __device__ vector operator*(var_type scalar) const 
    {
        return vector(x * scalar, y * scalar, z * scalar);
    }

    //cross product
    __device__ vector operator^(const vector& other) const 
    {
        return vector(  y * other.z - z * other.y,
                        z * other.x - x * other.z,
                        x * other.y - y * other.x );
    }
    
    //length
    __device__ var_type length() const
    {
        return sqrt(x*x+y*y+z*z);
    }


};

typedef struct vector vec3D;


/**
    * \brief Calculate the distance between two points
    *
    * @param x1 x coord. of point 1
    * @param y1 y coord. of point 1
    * @param z1 z coord. of point 1
    * @param x2 x coord. of point 2
    * @param y2 y coord. of point 2
    * @param z2 z coord. of point 2
    *
    * @return Distance between points
*/
__device__ inline var_type calculateDistance(var_type x1, var_type y1, var_type z1, var_type x2, var_type y2, var_type z2)
{
    var_type dx = x1-x2;
    var_type dy = y1-y2;
    var_type dz = z1-z2;
    return sqrt(dx*dx + dy*dy + dz*dz);
}

/**
    * \brief Calculates the normal component of vector v using the unit vector n (pointing from particle 1 to particle 2)
    *
    * @param v1 1st component of vector v
    * @param v2 2nd component of vector v
    * @param v3 3rd component of vector v
    * @param n1 1st component of vector n
    * @param n2 2nd component of vector n
    * @param n3 3rd component of vector n
    *
    * @return Returns the relative normal velocity
*/
__device__ inline var_type calculateNormal(var_type v1, var_type v2, var_type v3, var_type n1, var_type n2, var_type n3)
{
    return v1*n1 + v2*n2 + v3*n3;
}




#endif
