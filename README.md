# GPUDEM BETA
GPUDEM is a an open source GPU based discrete element solver using CUDA, currently in beta version.
The code is provided under the GNU GENERAL PUBLIC LICENSE.

# Installation
Download the repository, install CUDA (compute capability 8.0 or above required for full functionality) and a C++17 compiler. The code then can be compiled using the given makefile in the examples or the following command
```
nvcc -o GPUDEM -std=c++17 --gpu-architecture=sm_XX -maxrregcount=128 -I/path/to/GPUDEM
```
where XX is your CUDA compute capability.


# Usage
The code is documented using doxygen.
For usage refer to the provided examples. 

