
#include "utilities.h"
#include <assert.h>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <math.h>


string to_string(const double& num, const int& precision) {
	stringstream stream;
	stream << fixed << setprecision(precision) << num;
	return stream.str();
}

void initInput(double* H, double* HU, double* HV, double* H_ref, double* Zdx, double* Zdy,
	string filename, string solname, int grid_side, int nx)
{

	ifstream infile_H(filename + "_h.bin", ios::binary | ios::in);
	ifstream infile_HU(filename + "_hu.bin", ios::binary | ios::in);
	ifstream infile_HV(filename + "_hv.bin", ios::binary | ios::in);
	ifstream infile_H_ref(solname, ios::binary | ios::in);

	if (infile_H.is_open()) {
		for (int i(0); i < nx; ++i) {
			for (int j(0); j < nx; ++j) {
				infile_H.read(reinterpret_cast<char*>(&H[j * grid_side + i]), sizeof(double));
				infile_HU.read(reinterpret_cast<char*>(&HU[j * grid_side + i]), sizeof(double));
				infile_HV.read(reinterpret_cast<char*>(&HV[j * grid_side + i]), sizeof(double));
				infile_H_ref.read(reinterpret_cast<char*>(&H_ref[j * grid_side + i]), sizeof(double));
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
	if (infile_Zdx.is_open() & infile_Zdy.is_open()) {
		for (int i(0); i < nx; i++) {
			for (int j(0); j < nx; j++) {
				infile_Zdx.read(reinterpret_cast<char*>(&Zdx[j * grid_side + i]), sizeof(double));
				infile_Zdy.read(reinterpret_cast<char*>(&Zdy[j * grid_side + i]), sizeof(double));
			}
		}
	}
	else cout << "File not Open!\n";
	infile_Zdx.close();
	infile_Zdy.close();
}

void checkResults(double * H_ref, double * H, int grid_side, int nx)
{
	double avg_rel_err(0);
	for (int i(0); i < nx; ++i) {
		for (int j(0); j < nx; ++j) {
			double rel_error = abs(H_ref[i * grid_side + j] - H[i * grid_side + j]) / abs(H_ref[i * grid_side + j]);
			avg_rel_err += rel_error;
		}
	}
	std::cout << "Average relative error: " << avg_rel_err / (nx * nx) << endl;
};
