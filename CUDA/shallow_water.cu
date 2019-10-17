// std include
#include <fstream>  
#include <iostream>

// Local include
#include "utilities.h"
#include "calculations.cuh"


//--------------------------------------------------------------------------------------------------------------------------
// Final Code 
// The reference solution calculated by Matlab (Solution_nx2001_500km_T0.2_h_ref.bin & Solution_nx4001_500km_T0.2_h_ref.bin)
// are needed to be put in ./data folder for comparison.
//--------------------------------------------------------------------------------------------------------------------------

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
	if (result != cudaSuccess) {
		fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
		assert(result == cudaSuccess);
	}
#endif
	return result;
}

// host routine to set constant data
void setParameters()
{
	checkCuda(cudaMemcpyToSymbol(c_g, &g, sizeof(double), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(c_Tend, &Tend, sizeof(double), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(c_dx, &dx, sizeof(double), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(c_Size, &Size, sizeof(int), 0, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpyToSymbol(c_nx, &nx, sizeof(int), 0, cudaMemcpyHostToDevice));
}


int main(void)
{
	// Print device and precision
	cudaDeviceProp prop;
	checkCuda(cudaGetDeviceProperties(&prop, 0));
	printf("\nDevice Name: %s\n", prop.name);
	printf("Compute Capability: %d.%d\n\n", prop.major, prop.minor);
	setParameters();
	runTest();
	return 0;
}

void runTest()
{

	// Define chronometer
	float milliseconds;
	cudaEvent_t startEvent, stopEvent;
	checkCuda(cudaEventCreate(&startEvent));
	checkCuda(cudaEventCreate(&stopEvent));

	// Start the chronometer
	checkCuda(cudaEventRecord(startEvent, 0));

	// Define grid
	dim3 threads, threads_red, grid, grid_red;
	threads = dim3(BLOCK_WIDTH, BLOCK_HEIGHT, 1);
	grid = dim3(nx / threads.x + 1, nx / threads.y + 1, 1);

	// Define useful sizes
	int grid_side = grid.x * threads.x;
	int common_multiple = (nx / (BLOCK_HEIGHT * BLOCK_WIDTH) + 1) * BLOCK_HEIGHT * BLOCK_WIDTH;
	int vec_size = common_multiple * common_multiple;

	// Set filename
	string filename = "../data/Data_nx" + to_string(nx) + '_' + to_string(Size) + "km_T" + to_string(Tend, 1);
	string solname = "../data/Solution_nx" + to_string(nx) + '_' + to_string(Size) + "km_T" + to_string(Tend, 1) + "_h_ref.bin";

	// Initialize matrices
	double * H = new double[vec_size]();
	double * HU = new double[vec_size]();
	double * HV = new double[vec_size]();
	double * H_ref = new double[vec_size]();
	double * Zdx = new double[vec_size]();
	double * Zdy = new double[vec_size]();
	std::fill_n(H, vec_size, -10000);
	std::fill_n(H_ref, vec_size, -10000);

	// Read in data
	initInput(H, HU, HV, H_ref, Zdx, Zdy, filename, solname, grid_side, nx);

	// device arrays
	int bytes = vec_size * sizeof(double);
	double * d_H; double * d_HU; double * d_HV; double * d_Zdx; double * d_Zdy;

	//d_H, d_HU, d_HV have padding otherwise in comparison no rational number
	checkCuda(cudaMalloc((void**)& d_H, bytes));
	checkCuda(cudaMalloc((void**)& d_HU, bytes));
	checkCuda(cudaMalloc((void**)& d_HV, bytes));
	checkCuda(cudaMalloc((void**)& d_Zdx, bytes));
	checkCuda(cudaMalloc((void**)& d_Zdy, bytes));

	// Copy data from host to global memeory
	checkCuda(cudaMemcpy(d_H, H, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_HU, HU, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_HV, HV, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_Zdx, Zdx, bytes, cudaMemcpyHostToDevice));
	checkCuda(cudaMemcpy(d_Zdy, Zdy, bytes, cudaMemcpyHostToDevice));

	// Time related variables
	double T(0), nt(0), dt(0);
	double * d_T;
	checkCuda(cudaMalloc((void**)& d_T, sizeof(double)));

	// Allocate arrays used in calculation of nu
	double max_nu;
	double * d_nu_data;
	checkCuda(cudaMalloc((void**)& d_nu_data, sizeof(double) * vec_size));
	double * d_max_nu;
	checkCuda(cudaMalloc((void**)& d_max_nu, sizeof(double)));

	while (T < Tend) {

		// Set more shared memory for use of GPU
		cudaFuncSetCacheConfig(update, cudaFuncCachePreferShared);

		// Calculate max_nu and Perform the first reduction
		calculate_nu << <grid, threads >> > (d_H, d_HU, d_HV, d_nu_data);
		cudaDeviceSynchronize();

		max_reduction << <1024, 1024 >> > (d_nu_data, d_max_nu, vec_size);
		cudaDeviceSynchronize();

		// Transfer dT to kernels to set a correct time step before the end. 
		checkCuda(cudaMemcpy(d_T, &T, sizeof(double), cudaMemcpyHostToDevice));

		// Update of matrices in the kernel
		update << <grid, threads >> > (d_H, d_HU, d_HV, d_Zdx, d_Zdy, d_max_nu, d_T);
		cudaDeviceSynchronize();

		// Transfer dt from device to host
		checkCuda(cudaMemcpy(&max_nu, &d_max_nu[0], sizeof(double), cudaMemcpyDeviceToHost));
		dt = dx / (sqrt((double)2) * max_nu);

		if (T + dt > Tend) {
			dt = Tend - T;
		}

		// Print status
		std::cout << ("Computing T: " + to_string(T + dt) + ". " + to_string(100 * (T + dt) / Tend) + '%') << endl;

		// Update time T
		T = T + dt;
		nt = nt + 1;
	}


	// Transfer data back to host and check the result
	checkCuda(cudaMemcpy(H, d_H, bytes, cudaMemcpyDeviceToHost));
	checkResults(H_ref, H, grid_side, nx);

	// Print out the result
	string filename_output;
	filename_output = "Solution_nx" + to_string(nx) + '_' + to_string(Size) + "km" + "_T" + to_string(Tend, 1) + "_h_cuda.bin";
	ofstream outfile(filename_output, ios::out | ios::binary);
	for (int i(0); i < nx; ++i) {
		for (int j(0); j < nx; ++j) {
			outfile.write((char*)& H[j * grid_side + i], sizeof(double));
		}
	}
	outfile.close();

	checkCuda(cudaEventRecord(stopEvent, 0));
	checkCuda(cudaEventSynchronize(stopEvent));
	checkCuda(cudaEventElapsedTime(&milliseconds, startEvent, stopEvent));

	printf("   Average time (ms): %f\n", milliseconds);
	printf("   Average Bandwidth (GB/s): %f\n\n",
		nt * (15 + 2 + 11 + 30 + 30 + 1) * pow(nx, 2) / (pow(10, 6) * milliseconds));

	checkCuda(cudaEventDestroy(startEvent));
	checkCuda(cudaEventDestroy(stopEvent));

	checkCuda(cudaFree(d_H));
	checkCuda(cudaFree(d_HU));
	checkCuda(cudaFree(d_HV));
	checkCuda(cudaFree(d_Zdx));
	checkCuda(cudaFree(d_Zdy));
	checkCuda(cudaFree(d_nu_data));
	checkCuda(cudaFree(d_max_nu));
	checkCuda(cudaFree(d_T));

	delete[] H;
	delete[] HU;
	delete[] H_ref;
	delete[] HV;
	delete[] Zdx;
	delete[] Zdy;
}



