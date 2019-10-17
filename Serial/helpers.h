//
// Created by Jiahua WU on 11.05.19.
//

#ifndef SHALLOW_WATER_HELPERS_H
#define SHALLOW_WATER_HELPERS_H

#include <string>
using namespace std;
string to_string(const double& num, const int& precision);
void loaddata(double** H, double** HU, double** HV, double** H_ref, double** Zdx, double** Zdy, int nx, string filename, string solname);
double calculate_max_nu(double** H, double** HU, double** HV, double g, int nx);
void update(double** H, double** HU, double** HV, double** Ht, double** HUt, double** HVt, double** Zdx, double** Zdy, int nx, double dt, double dx, double g);
void checkResults(double** H_ref, double** H, int nx);
#endif //SHALLOW_WATER_HELPERS_H
