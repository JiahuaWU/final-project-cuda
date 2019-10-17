#ifndef CALCULATION_H
#define CALCULATION_H

// CUDA include
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_functions.h"
#include "device_launch_parameters.h"
#include "math_functions.h"
#include "utilities.h"
using namespace std;

#define BLOCK_WIDTH 16 //length in x direction(horizontal direction)
#define BLOCK_HEIGHT 12 //length in y direction(vertical direction)

// Device constants
__constant__ double c_g, c_Tend, c_dx;
__constant__ int c_Size, c_nx;

// Calculate the parameter mu used in calculation of new time-step and Perform the first reduction
__global__ void calculate_nu(double* H, double* HU, double* HV, double* results);

// Perform max reduction on input and output to results
__global__ void max_reduction(double* in, double* out, int N);

// Perform simulation-time step 
__global__ void update(double* H, double* HU, double* HV, double* Zdx, double* Zdy, double* max_mu, double* d_T);

// Reducemax function in a wrap
__inline__ __device__
double warpReduceMax(double val);

// Reducemax funciton in a block
__inline__ __device__
double blockReduceMax(double val);

// Atomic max function for maxreduce
__device__ double atomicMax(double* address, double val);

//--------------------------------------------------------------------------------------
// Kernel Definition
//--------------------------------------------------------------------------------------

__global__ void calculate_nu(double* H, double* HU, double* HV, double* results)
{
	double nu;
	int global_j = blockIdx.x * blockDim.x + threadIdx.x;
	int global_i = blockIdx.y * blockDim.y + threadIdx.y;
	int grid_width = gridDim.x * blockDim.x;

	// Calculate nu for non-padding elements
	if (H[global_i * grid_width + global_j] > 0) {
		double a = HU[global_i * grid_width + global_j] / H[global_i * grid_width + global_j];
		double b = HV[global_i * grid_width + global_j] / H[global_i * grid_width + global_j];
		double c = sqrt(H[global_i * grid_width + global_j] * c_g);
		double r1 = fmax(fabs(a - c), fabs(a + c));
		double r2 = fmax(fabs(b - c), fabs(b + c));
		nu = sqrt(r1 * r1 + r2 * r2);
	}
	else nu = -10000;
	results[global_i * grid_width + global_j] = nu;
}

__global__ void update(double * H, double * HU, double * HV, double * Zdx, double * Zdy, double * max_nu, double * d_T)
{
	//2 - wide halo(each update needs 1 boundary data of the neighbour in each direction) 
	__shared__ double s_H[BLOCK_HEIGHT + 2][BLOCK_WIDTH + 2], s_HU[BLOCK_HEIGHT + 2][BLOCK_WIDTH + 2],
		s_HV[BLOCK_HEIGHT + 2][BLOCK_WIDTH + 2], s_Ht[BLOCK_HEIGHT + 2][BLOCK_WIDTH + 2],
		s_HUt[BLOCK_HEIGHT + 2][BLOCK_WIDTH + 2], s_HVt[BLOCK_HEIGHT + 2][BLOCK_WIDTH + 2];

	// Useful indices
	int global_i = blockIdx.y * blockDim.y + threadIdx.y;
	int global_j = blockIdx.x * blockDim.x + threadIdx.x;
	int grid_width = gridDim.x * blockDim.x;
	int grid_height = gridDim.y * blockDim.y;
	int len_padding_width = grid_width - c_nx;
	int len_padding_height = grid_height - c_nx;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int s_i = ty + 1; //local i for shared memory access + halo offset
	int s_j = tx + 1; //local j for shared memory access + halo offset

	// Constant
	double T = d_T[0];
	double dt = c_dx / (sqrt((double)2) * max_nu[0]);
	if (T + dt > c_Tend) {
		dt = c_Tend - T;
	}
	double C = (0.5 * dt / c_dx);

	// Read in elements treated by the nodes in the grid
	s_H[s_i][s_j] = H[global_i * grid_width + global_j];
	s_HU[s_i][s_j] = HU[global_i * grid_width + global_j];
	s_HV[s_i][s_j] = HV[global_i * grid_width + global_j];
	s_Ht[s_i][s_j] = H[global_i * grid_width + global_j];
	s_HUt[s_i][s_j] = HU[global_i * grid_width + global_j];
	s_HVt[s_i][s_j] = HV[global_i * grid_width + global_j];

	__syncthreads();

	// Read in halo data
	if (tx == 0 & global_j > 0) {
		s_H[s_i][0] = H[global_i * grid_width + global_j - 1];
		s_HU[s_i][0] = HU[global_i * grid_width + global_j - 1];
		s_HV[s_i][0] = HV[global_i * grid_width + global_j - 1];
		s_Ht[s_i][0] = H[global_i * grid_width + global_j - 1];
		s_HUt[s_i][0] = HU[global_i * grid_width + global_j - 1];
		s_HVt[s_i][0] = HV[global_i * grid_width + global_j - 1];
	}

	if (tx == BLOCK_WIDTH - 1 & global_j < c_nx - 1) {
		s_H[s_i][BLOCK_WIDTH + 1] = H[global_i * grid_width + global_j + 1];
		s_HU[s_i][BLOCK_WIDTH + 1] = HU[global_i * grid_width + global_j + 1];
		s_HV[s_i][BLOCK_WIDTH + 1] = HV[global_i * grid_width + global_j + 1];
		s_HUt[s_i][BLOCK_WIDTH + 1] = HU[global_i * grid_width + global_j + 1];
		s_HVt[s_i][BLOCK_WIDTH + 1] = HV[global_i * grid_width + global_j + 1];
		s_Ht[s_i][BLOCK_WIDTH + 1] = H[global_i * grid_width + global_j + 1];
	}

	if (ty == 0 & global_i > 0) {
		s_H[0][s_j] = H[(global_i - 1) * grid_width + global_j];
		s_HU[0][s_j] = HU[(global_i - 1) * grid_width + global_j];
		s_HV[0][s_j] = HV[(global_i - 1) * grid_width + global_j];
		s_Ht[0][s_j] = H[(global_i - 1) * grid_width + global_j];
		s_HUt[0][s_j] = HU[(global_i - 1) * grid_width + global_j];
		s_HVt[0][s_j] = HV[(global_i - 1) * grid_width + global_j];
	}

	if (ty == BLOCK_HEIGHT - 1 & global_i < c_nx - 1) {
		s_H[BLOCK_HEIGHT + 1][s_j] = H[(global_i + 1) * grid_width + global_j];
		s_HU[BLOCK_HEIGHT + 1][s_j] = HU[(global_i + 1) * grid_width + global_j];
		s_HV[BLOCK_HEIGHT + 1][s_j] = HV[(global_i + 1) * grid_width + global_j];
		s_Ht[BLOCK_HEIGHT + 1][s_j] = H[(global_i + 1) * grid_width + global_j];
		s_HUt[BLOCK_HEIGHT + 1][s_j] = HU[(global_i + 1) * grid_width + global_j];
		s_HVt[BLOCK_HEIGHT + 1][s_j] = HV[(global_i + 1) * grid_width + global_j];
	}

	__syncthreads();

	// Apply Ht, HUt, HVt boundary condition
	if (global_j == 0) {
		s_Ht[s_i][1] = s_Ht[s_i][2];
		s_HUt[s_i][1] = s_HUt[s_i][2];
		s_HVt[s_i][1] = s_HVt[s_i][2];
	}

	if (global_j == 0 & tx == 0 & ty == 0) {
		s_Ht[0][1] = s_Ht[0][2];
		s_HUt[0][1] = s_HUt[0][2];
		s_HVt[0][1] = s_HVt[0][2];
		s_Ht[BLOCK_HEIGHT + 1][1] = s_Ht[BLOCK_HEIGHT + 1][2];
		s_HUt[BLOCK_HEIGHT + 1][1] = s_HUt[BLOCK_HEIGHT + 1][2];
		s_HVt[BLOCK_HEIGHT + 1][1] = s_HVt[BLOCK_HEIGHT + 1][2];
	}

	if (global_j == c_nx - 1) {
		s_Ht[s_i][BLOCK_WIDTH - len_padding_width] = s_Ht[s_i][BLOCK_WIDTH - len_padding_width - 1];
		s_HUt[s_i][BLOCK_WIDTH - len_padding_width] = s_HUt[s_i][BLOCK_WIDTH - len_padding_width - 1];
		s_HVt[s_i][BLOCK_WIDTH - len_padding_width] = s_HVt[s_i][BLOCK_WIDTH - len_padding_width - 1];
	}

	if (global_j == c_nx - 1 & tx == BLOCK_WIDTH - len_padding_width - 1 & ty == BLOCK_HEIGHT - len_padding_height - 1) {
		s_Ht[0][BLOCK_WIDTH - len_padding_width] = s_Ht[0][BLOCK_WIDTH - len_padding_width - 1];
		s_HUt[0][BLOCK_WIDTH - len_padding_width] = s_HUt[0][BLOCK_WIDTH - len_padding_width - 1];
		s_HVt[0][BLOCK_WIDTH - len_padding_width] = s_HVt[0][BLOCK_WIDTH - len_padding_width - 1];
		s_Ht[BLOCK_HEIGHT + 1][BLOCK_WIDTH - len_padding_width] = s_Ht[BLOCK_HEIGHT + 1][BLOCK_WIDTH - len_padding_width - 1];
		s_HUt[BLOCK_HEIGHT + 1][BLOCK_WIDTH - len_padding_width] = s_HUt[BLOCK_HEIGHT + 1][BLOCK_WIDTH - len_padding_width - 1];
		s_HVt[BLOCK_HEIGHT + 1][BLOCK_WIDTH - len_padding_width] = s_HVt[BLOCK_HEIGHT + 1][BLOCK_WIDTH - len_padding_width - 1];
	}

	__syncthreads();

	if (global_i == 0 & global_j != 0) {
		s_Ht[1][s_j] = s_Ht[2][s_j];
		s_HUt[1][s_j] = s_HUt[2][s_j];
		s_HVt[1][s_j] = s_HVt[2][s_j];
	}

	if (global_i == 0 & tx == 0 & ty == 0) {
		s_Ht[1][0] = s_Ht[2][0];
		s_HUt[1][0] = s_HUt[2][0];
		s_HVt[1][0] = s_HVt[2][0];
		s_Ht[1][BLOCK_WIDTH + 1] = s_Ht[2][BLOCK_WIDTH + 1];
		s_HUt[1][BLOCK_WIDTH + 1] = s_HUt[2][BLOCK_WIDTH + 1];
		s_HVt[1][BLOCK_WIDTH + 1] = s_HVt[2][BLOCK_WIDTH + 1];
	}

	if (global_i == c_nx - 1 & global_j != c_nx - 1) {
		s_Ht[BLOCK_HEIGHT - len_padding_height][s_j] = s_Ht[BLOCK_HEIGHT - len_padding_height - 1][s_j];
		s_HUt[BLOCK_HEIGHT - len_padding_height][s_j] = s_HUt[BLOCK_HEIGHT - len_padding_height - 1][s_j];
		s_HVt[BLOCK_HEIGHT - len_padding_height][s_j] = s_HVt[BLOCK_HEIGHT - len_padding_height - 1][s_j];
	}

	if (global_i == c_nx - 1 & tx == BLOCK_WIDTH - len_padding_width - 1 & ty == BLOCK_HEIGHT - len_padding_height - 1) {
		s_Ht[BLOCK_HEIGHT - len_padding_height][0] = s_Ht[BLOCK_HEIGHT - len_padding_height - 1][0];
		s_HUt[BLOCK_HEIGHT - len_padding_height][0] = s_HUt[BLOCK_HEIGHT - len_padding_height - 1][0];
		s_HVt[BLOCK_HEIGHT - len_padding_height][0] = s_HVt[BLOCK_HEIGHT - len_padding_height - 1][0];
		s_Ht[BLOCK_HEIGHT - len_padding_height][BLOCK_WIDTH + 1] = s_Ht[BLOCK_HEIGHT - len_padding_height - 1][BLOCK_WIDTH + 1];
		s_HUt[BLOCK_HEIGHT - len_padding_height][BLOCK_WIDTH + 1] = s_HUt[BLOCK_HEIGHT - len_padding_height - 1][BLOCK_WIDTH + 1];
		s_HVt[BLOCK_HEIGHT - len_padding_height][BLOCK_WIDTH + 1] = s_HVt[BLOCK_HEIGHT - len_padding_height - 1][BLOCK_WIDTH + 1];
	}

	__syncthreads();

	// Update interior points
	if (global_i > 0 && global_i < c_nx - 1 && global_j > 0 && global_j < c_nx - 1) {
		s_H[s_i][s_j] = 0;
		s_H[s_i][s_j] += 0.25 * (s_Ht[s_i][s_j - 1] + s_Ht[s_i][s_j + 1] + s_Ht[s_i - 1][s_j] + s_Ht[s_i + 1][s_j]);
		s_H[s_i][s_j] += C * (s_HUt[s_i][s_j - 1] - s_HUt[s_i][s_j + 1] + s_HVt[s_i - 1][s_j] - s_HVt[s_i + 1][s_j]);

		s_HU[s_i][s_j] = 0;
		s_HU[s_i][s_j] += 0.25 * (s_HUt[s_i][s_j - 1] + s_HUt[s_i][s_j + 1] + s_HUt[s_i - 1][s_j] + s_HUt[s_i + 1][s_j]);

		s_HU[s_i][s_j] -= dt * c_g * s_H[s_i][s_j] * Zdx[global_i*grid_width+global_j];
		s_HU[s_i][s_j] += C * (s_HUt[s_i][s_j - 1] * s_HUt[s_i][s_j - 1] / s_Ht[s_i][s_j - 1]
			+ 0.5 * c_g * s_Ht[s_i][s_j - 1] * s_Ht[s_i][s_j - 1] - s_HUt[s_i][s_j + 1] * s_HUt[s_i][s_j + 1] / s_Ht[s_i][s_j + 1]
			- 0.5 * c_g * s_Ht[s_i][s_j + 1] * s_Ht[s_i][s_j + 1]);
		s_HU[s_i][s_j] += C * (s_HUt[s_i - 1][s_j] * s_HVt[s_i - 1][s_j] / s_Ht[s_i - 1][s_j] -
			s_HUt[s_i + 1][s_j] * s_HVt[s_i + 1][s_j] / s_Ht[s_i + 1][s_j]);

		s_HV[s_i][s_j] = 0;
		s_HV[s_i][s_j] += 0.25 * (s_HVt[s_i][s_j - 1] + s_HVt[s_i][s_j + 1] + s_HVt[s_i - 1][s_j] + s_HVt[s_i + 1][s_j]);
		s_HV[s_i][s_j] -= dt * c_g * s_H[s_i][s_j] * Zdy[global_i * grid_width + global_j];
		s_HV[s_i][s_j] += C * (s_HUt[s_i][s_j - 1] * s_HVt[s_i][s_j - 1] / s_Ht[s_i][s_j - 1] - s_HUt[s_i][s_j + 1] * s_HVt[s_i][s_j + 1] / s_Ht[s_i][s_j + 1]);
		s_HV[s_i][s_j] += C * (s_HVt[s_i - 1][s_j] * s_HVt[s_i - 1][s_j] / s_Ht[s_i - 1][s_j] + 0.5 * c_g * s_Ht[s_i - 1][s_j] * s_Ht[s_i - 1][s_j]
			- s_HVt[s_i + 1][s_j] * s_HVt[s_i + 1][s_j] / s_Ht[s_i + 1][s_j] - 0.5 * c_g * s_Ht[s_i + 1][s_j] * s_Ht[s_i + 1][s_j]);

		if (s_H[s_i][s_j] < 0) s_H[s_i][s_j] = 0.00001;
		if (s_H[s_i][s_j] <= 0.0005) s_HU[s_i][s_j] = 0.;
		if (s_H[s_i][s_j] <= 0.0005) s_HV[s_i][s_j] = 0.;
	}

	__syncthreads();

	// Write solution for sharing among blocks
	H[global_i * grid_width + global_j] = s_H[s_i][s_j];
	HU[global_i * grid_width + global_j] = s_HU[s_i][s_j];
	HV[global_i * grid_width + global_j] = s_HV[s_i][s_j];
}

__global__ void max_reduction(double * in, double * out, int N) {

	double max = double(-100);
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < N;
		i += blockDim.x * gridDim.x) {
		max = fmax(max, in[i]);
	}
	max = blockReduceMax(max);
	if (threadIdx.x == 0)
		atomicMax(out, max);
}

__inline__ __device__
double warpReduceMax(double val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2)
		val = fmax(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
	return val;
}

__inline__ __device__

double blockReduceMax(double val) {
	static __shared__ double shared[32]; // Shared mem for 32 partial maxs
	int tid = threadIdx.x + threadIdx.y * blockDim.x;
	int lane = tid % warpSize;
	int wid = tid / warpSize;

	val = warpReduceMax(val);     // Each warp performs partial reduction
	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory
	__syncthreads();              // Wait for all partial reductions
	//read from shared memory only if that warp existed
	val = (tid < blockDim.x * blockDim.y / warpSize) ? shared[lane] : 0;
	if (wid == 0) val = warpReduceMax(val); //Final reduce within first warp
	return val;
}


__device__ double atomicMax(double* address, double val)
{
	unsigned long long int* address_as_ull =
		(unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;

	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed,
			__double_as_longlong(fmax(val,
				__longlong_as_double(assumed))));
		// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
	} while (assumed != old);
	return __longlong_as_double(old);
}

#endif // CALCULATION_H
