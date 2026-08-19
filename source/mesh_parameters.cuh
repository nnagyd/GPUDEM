/**
 * @file mesh_parameters.cuh
 * @brief Runtime mesh parameters passed to the device solver.
 *
 * RuntimeMeshParameters mirrors the compile-time DecomposedDomainsConstants that
 * each simulation translation unit (example_*.cu) defines before including the
 * solver headers. Linked-cell buffers stay allocated with those compile-time
 * maxima; runtime values must remain within them.
 */

#ifndef mesh_parameters_H
#define mesh_parameters_H

#include "settings.cuh"

struct RuntimeMeshParameters
{
    int nx = DecomposedDomainsConstants::Nx;
    int ny = DecomposedDomainsConstants::Ny;
    int nz = DecomposedDomainsConstants::Nz;
    int ncell = DecomposedDomainsConstants::Ncell;

    var_type minx = DecomposedDomainsConstants::minx;
    var_type miny = DecomposedDomainsConstants::miny;
    var_type minz = DecomposedDomainsConstants::minz;
    var_type maxx = DecomposedDomainsConstants::maxx;
    var_type maxy = DecomposedDomainsConstants::maxy;
    var_type maxz = DecomposedDomainsConstants::maxz;

    var_type NoverDx = DecomposedDomainsConstants::NoverDx;
    var_type NoverDy = DecomposedDomainsConstants::NoverDy;
    var_type NoverDz = DecomposedDomainsConstants::NoverDz;
};

inline RuntimeMeshParameters defaultRuntimeMeshParameters()
{
    return RuntimeMeshParameters{};
}

#endif
