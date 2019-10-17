# Shallow Water CUDA-PHPC-2019-EPFL

### Introduction

This project is about the application of CUDA to a shallow water equation used for simulation of tunami. I

In this project, I created 2 different codes:
- Serial version of the shallow water equation
- Parallelized version using CUDA

The purpose of this project was to test the serial version and to transfer them into parallelized version using CUDA.

### Sturcture of the CDUA code

- shallow_water.cu: The main program where the function runTest() is executed. 
- utilities.h: Definition of physical constants (**g**) and the discretization constants (**nx**, **dx**,etc.) and definition of helper functions used in the main program.
- utilities.cpp: Source file of utilities.h.
- calculations.cuh: Definition of the kernels used in calculations and Definition of the block width **BLOCK_WEIGHT** and block height **BLOCK_HEIGHT** used in calculation. 

### Compiling the C++ code
The serial code can be complied via cmake. The CUDA code is developped using CUDA Toolkit 10.1 and can be complied with NVCC complier.

### Executing the C++ code
The data path is set to be a folder called data at the path of the executable. If there is no such a folder, there will be an error message "File not open!" when running the program. 
It would be better that the folder contains a referenced solution (of the form "Solution_nx4001_500km_T0.2_h_ref.bin") which will be read in and compared with the result. One can
adjust **nx** to run the program for different solution






