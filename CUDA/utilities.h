#ifndef UTILITES_H
#define UTILITES_H

#include "cuda_runtime_api.h"
#include <string>
using namespace std;

// Definition of constants
const double g = 127267.2000000000000000;    //% Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
const int Size = 500;              //% Size of map, Size*Size [km]
const int nx = 2001;             //% Number of cells in each direction on the grid
const double Tend = 0.20;             //% Simulation time in hours [hr]
const double dx = (double)Size / (double)nx;          //% Grid spacening

// Transfer a Real to a string for a given precision (used in define read in filename)
string to_string(const double& num, const int& precision);

// Read in data from ./data and initialize data arrays
void initInput(double* H, double* HU, double* HV, double* H_ref, double* Zdx, double* Zdy,
	string filename, string solname, int grid_side, int nx);

// Check result by comparing with the one obtained by matlab 
void checkResults(double* H_ref, double* H, int grid_side, int nx);

// Run the whole process and check the result
void runTest();

#endif
