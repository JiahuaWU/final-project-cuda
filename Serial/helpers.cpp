//
// Created by Jiahua WU on 11.05.19.
//

#include "helpers.h"
#include <string>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <math.h>
using namespace std;
string to_string(const double& num, const int& precision){
    stringstream stream;
    stream << fixed << setprecision(precision) << num;
    return stream.str();
};

void loaddata(double** H, double** HU, double** HV, double** H_ref, double** Zdx, double** Zdy, int nx, string filename, string solname)
{
	ifstream infile_H(filename + "_h.bin", ios::binary | ios::in);
	ifstream infile_HU(filename + "_hu.bin", ios::binary | ios::in);
	ifstream infile_HV(filename + "_hv.bin", ios::binary | ios::in);
	ifstream infile_H_ref(solname, ios::binary | ios::in);

	if (infile_H.is_open()) {
		for (int i(0); i < nx; i++) {
			for (int j(0); j < nx; j++) {
				infile_H.read(reinterpret_cast<char*>(&H[j][i]), sizeof(double));
				infile_HU.read(reinterpret_cast<char*>(&HU[j][i]), sizeof(double));
				infile_HV.read(reinterpret_cast<char*>(&HV[j][i]), sizeof(double));
				infile_H_ref.read(reinterpret_cast<char*>(&H_ref[j][i]), sizeof(double));
			}
		}
	}
	else std::cout << "File not open!\n";
	infile_H.close();
	infile_HU.close();
	infile_HV.close();
	infile_H_ref.close();

	//    % Load topography slopes from data files

	ifstream infile_Zdx(filename + "_Zdx.bin", ios::binary | ios::in);
	ifstream infile_Zdy(filename + "_Zdy.bin", ios::binary | ios::in);
	for (int i(0); i < nx; i++) {
		for (int j(0); j < nx; j++) {
			infile_Zdx.read(reinterpret_cast<char*>(&Zdx[j][i]), sizeof(double));
			infile_Zdy.read(reinterpret_cast<char*>(&Zdy[j][i]), sizeof(double));
		}
	}
	infile_Zdx.close();
	infile_Zdy.close();
}

double calculate_max_nu(double** H, double** HU, double** HV, double g, int nx)
{
	double max_nu = LONG_MIN;
	for (int i(0); i < nx; ++i) {
		for (int j(0); j < nx; ++j) {
			double a = HU[i][j] / H[i][j];
			double b = HV[i][j] / H[i][j];
			double c = sqrt(H[i][j] * g);
			double r1 = max(abs(a - c), abs(a + c));
			double r2 = max(abs(b - c), abs(b + c));
			double mu = sqrt(r1 * r1 + r2 * r2);
			max_nu = max(max_nu, sqrt(r1 * r1 + r2 * r2));
		}
	}
	return max_nu;
}

void update(double** H, double** HU, double** HV, double** Ht, double** HUt, double** HVt, double** Zdx, double** Zdy, int nx,double dt, double dx, double g)
{
	//        %% Copy solution to temp storage and enforce boundary condition

	for (int i(0); i < nx; i++) {
		for (int j(0); j < nx; j++) {
			Ht[i][j] = H[i][j];
			HUt[i][j] = HU[i][j];
			HVt[i][j] = HV[i][j];
		}
	}

	int end = nx - 1;
	for (int i(0); i < nx; ++i) {
		Ht[0][i] = Ht[1][i];
		Ht[end][i] = Ht[end - 1][i];
		Ht[i][0] = Ht[i][1];
		Ht[i][end] = Ht[i][end - 1];

		HUt[0][i] = HUt[1][i];
		HUt[end][i] = HUt[end - 1][i];
		HUt[i][0] = HUt[i][1];
		HUt[i][end] = HUt[i][end - 1];

		HVt[0][i] = HVt[1][i];
		HVt[end][i] = HVt[end - 1][i];
		HVt[i][0] = HVt[i][1];
		HVt[i][end] = HVt[i][end - 1];
	}


	//       %% Compute a time - step

	double C = (0.5 * dt / dx);
	double c1 = 0.5 * g;
	double c2 = dt * g;
	for (int i(1); i < nx - 1; ++i) {
		for (int j(1); j < nx - 1; ++j) {
			H[i][j] = 0;
			H[i][j] += 0.25 * (Ht[i][j - 1] + Ht[i][j + 1] + Ht[i - 1][j] + Ht[i + 1][j]);
			H[i][j] += C * (HUt[i][j - 1] - HUt[i][j + 1] + HVt[i - 1][j] - HVt[i + 1][j]);

			HU[i][j] = 0;
			HU[i][j] += 0.25 * (HUt[i][j - 1] + HUt[i][j + 1] + HUt[i - 1][j] + HUt[i + 1][j]);
			HU[i][j] -= c2 * H[i][j] * Zdx[i][j];
			HU[i][j] += C * (HUt[i][j - 1] * HUt[i][j - 1] / Ht[i][j - 1]
				+ c1 * Ht[i][j - 1] * Ht[i][j - 1] - HUt[i][j + 1] * HUt[i][j + 1] / Ht[i][j + 1]
				- c1 * Ht[i][j + 1] * Ht[i][j + 1]);
			HU[i][j] += C * (HUt[i - 1][j] * HVt[i - 1][j] / Ht[i - 1][j] -
				HUt[i + 1][j] * HVt[i + 1][j] / Ht[i + 1][j]);

			HV[i][j] = 0;
			HV[i][j] += 0.25 * (HVt[i][j - 1] + HVt[i][j + 1] + HVt[i - 1][j] + HVt[i + 1][j]);
			HV[i][j] -= c2 * H[i][j] * Zdy[i][j];
			HV[i][j] += C * (HUt[i][j - 1] * HVt[i][j - 1] / Ht[i][j - 1] - HUt[i][j + 1] * HVt[i][j + 1] / Ht[i][j + 1]);
			HV[i][j] += C * (HVt[i - 1][j] * HVt[i - 1][j] / Ht[i - 1][j] + c1 * Ht[i - 1][j] * Ht[i - 1][j]
				- HVt[i + 1][j] * HVt[i + 1][j] / Ht[i + 1][j] - c1 * Ht[i + 1][j] * Ht[i + 1][j]);
		}
	}

	for (int i(0); i < nx; ++i) {
		for (int j(0); j < nx; ++j) {
			if (H[i][j] < 0) H[i][j] = 0.00001;
			if (H[i][j] <= 0.0005) HU[i][j] = 0.;
			if (H[i][j] <= 0.0005) HV[i][j] = 0.;
		}
	}
}

void checkResults(double** H_ref, double** H,int nx)
{
	double max_rel_err(0);
	for (int i(0); i < nx; ++i) {
		for (int j(0); j < nx; ++j) {
			double error = abs(H_ref[i][j] - H[i][j]) / abs(H_ref[i][j]);
			if (error > max_rel_err) {
				max_rel_err = error;
			}
		}
	}
	std::cout << "Maximum relative error: " << max_rel_err << endl;
}