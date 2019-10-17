

#include <iostream>
#include <algorithm>
#include <math.h>
#include <string>
#include <fstream>
#include "helpers.h"
#include <ctime>

//-------------------------------------------------------------------------------------------------------------------------
// The reference solution caculated by Matlab (Solution_nx2001_500km_T0.2_h_ref.bin & Solution_nx4001_500km_T0.2_h_ref.bin)
// are needed to be put in ./data folder for comparison.
//-------------------------------------------------------------------------------------------------------------------------

using namespace std;
int main() {
	string data_path, filename, solname;
	double probe1, probe2, probe3;
	//    %% Load initial condition from disk

	const double& g = 127267.2000000000000000;    //% Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
	const int& Size = 500;              //% Size of map, Size*Size [km]
	const int& nx = 2001;             //% Number of cells in each direction on the grid
	const double& Tend = 0.20;             //% Simulation time in hours [hr]
	const double& dx = (double)Size / (double)nx;          //% Grid spacening
	

//    %% Add data path and start timer
	data_path = "./data/";

	//% Set filename
	filename = data_path+"Data_nx" + to_string(nx) + '_' + to_string(Size) + "km_T" + to_string(Tend, 1);
	solname  = data_path+"Solution_nx" + to_string(nx) + '_' + to_string(Size) + "km_T" + to_string(Tend, 1) +"_h_ref.bin";

	clock_t t_start_allocation;
	clock_t t_start_input;
	clock_t t_start_calculation;
	clock_t t_start_output;

	t_start_allocation = clock();
	//    % Load initial condition from data files
	double** H = new double* [nx];
	double** HU = new double* [nx];
	double** HV = new double* [nx];
	double** H_ref = new double* [nx];

	for (int i(0); i < nx; ++i) {
		H[i] = new double[nx];
		HU[i] = new double[nx];
		HV[i] = new double[nx];
		H_ref[i] = new double[nx];
	}
	double** Zdx = new double* [nx];
	double** Zdy = new double* [nx];

	for (int i(0); i < nx; ++i) {
		Zdx[i] = new double[nx];
		Zdy[i] = new double[nx];
	}

	double** Ht = new double* [nx];
	double** HUt = new double* [nx];
	double** HVt = new double* [nx];

	for (int i(0); i < nx; ++i) {
		Ht[i] = new double[nx];
		HUt[i] = new double[nx];
		HVt[i] = new double[nx];
	}


	double time_allocation = (clock() - t_start_allocation) / (double)CLOCKS_PER_SEC;

	t_start_input = clock();

	loaddata(H, HU, HV, H_ref, Zdx, Zdy, nx, filename, solname);


	double time_input = (clock() - t_start_input) / (double)CLOCKS_PER_SEC;



	//    %% Compute all time-steps

	//  %% Compute the time-step length
	double T(0), nt(0), dt, max_nu;

	t_start_calculation = clock();
	while (T < Tend) {
		max_nu = calculate_max_nu(H,HU,HV, g, nx);
//		cout << "max mu: " << max_mu << endl;
		dt = dx / (sqrt(2) * max_nu);
		

		//        mu = sqrt(max(abs(HU./ H - sqrt(H * g)), abs(HU./ H + sqrt(H * g))).^
		//                  2 + max(abs(HV./ H - sqrt(H * g)), abs(HV./ H + sqrt(H * g))).^ 2);
		//        dt = dx / (sqrt(2) * max(mu(:)));
		if (T + dt > Tend) {
			dt = Tend - T;
		}

		//        % % Print status

		std::cout << ("Computing T: " + to_string(T + dt) + ". " + to_string(100 * (T + dt) / Tend) + '%') << endl;

		update(H, HU, HV, Ht, HUt, HVt, Zdx, Zdy, nx, dt, dx, g);

		//     %% Update time T
		T = T + dt;
		nt = nt + 1;
	}
	double calculation_time = (clock() - t_start_calculation) / (double)CLOCKS_PER_SEC;




	//        H(2:nx - 1, 2:nx - 1, 1) = 0.25 * (Ht(2:nx - 1, 1:nx - 2)+Ht(2:nx - 1, 3:nx)+Ht(1:nx - 2, 2:nx - 1)+Ht(3:nx, 2:nx - 1)) ...
	//                                   + C * (HUt(2:nx - 1, 1:nx - 2) -HUt(2:nx - 1, 3:nx) +HVt(1:nx - 2, 2:nx - 1) -HVt(3:nx, 2:nx - 1));

	//        HU(2:nx - 1, 2:nx - 1) = 0.25 * (HUt(2:nx - 1, 1:nx - 2)+HUt(2:nx - 1, 3:nx)+HUt(1:nx - 2, 2:nx - 1)+HUt(3:nx, 2:nx - 1))
	//                                 -dt * g * H(2:nx - 1, 2:nx - 1).*Zdx(2:nx - 1, 2:nx - 1) +C * (HUt(2:nx - 1, 1:nx - 2).^ 2. / Ht(2:nx - 1, 1:nx - 2)
	//                                  +0.5 * g * Ht(2:nx - 1, 1:nx - 2).^ 2 - HUt(2:nx - 1, 3:nx).^ 2. / Ht(2:nx - 1, 3:nx)
	//                                  -0.5 * g * Ht(2:nx - 1, 3:nx).^ 2 )
	//                                  +C * (HUt(1:nx - 2, 2:nx - 1).*HVt(1:nx - 2, 2:nx - 1)./Ht(1:nx - 2, 2:nx - 1) -HUt(3:nx, 2:nx - 1).*HVt( 3:nx, 2:nx - 1)./Ht(3:nx, 2:nx - 1));

	//        HV(2:nx - 1, 2:nx - 1) = 0.25 * (HVt(2:nx - 1, 1:nx - 2)+HVt(2:nx - 1, 3:nx)+HVt(1:nx - 2, 2:nx - 1)+HVt(3:nx, 2:nx - 1))
	//                                 -dt * g * H(2:nx - 1, 2:nx - 1).*Zdy(2:nx - 1, 2:nx - 1)
	//                                 + C * (HUt(2:nx - 1, 1:nx - 2).*HVt(2:nx - 1, 1:nx - 2)./Ht(2:nx - 1, 1:nx - 2) - HUt(2:nx - 1, 3:nx).*HVt(2:nx - 1, 3:nx)./Ht(2:nx - 1, 3:nx))
	//                                   + C * (HVt(1:nx - 2, 2:nx - 1).^ 2. / Ht(1:nx - 2, 2:nx - 1) + 0.5 * g * Ht(1:nx - 2, 2:nx - 1).^ 2
	//                                   - HVt(3:nx, 2:nx - 1).^ 2. / Ht(3:nx, 2:nx - 1) -0.5 * g * Ht(3:nx, 2:nx - 1).^ 2  );

	//        %% Impose tolerances
	//
	//        H(H < 0) = 0.00001;
	//        HU(H <= 0.0005) = 0;
	//        HV(H <= 0.0005) = 0;

	t_start_output = clock();

	checkResults(H_ref,H,nx);

	//    %% Save solution to disk
	filename = "Solution_nx" + to_string(nx) + '_' + to_string(Size) + "km" + "_T" + to_string(Tend, 1) + "_h_serial.bin";
	ofstream outfile(filename, ios::out | ios::binary);
	for (int i(0); i < nx; ++i)
		for (int j(0); j < nx; ++j)
			outfile.write((char*)& H[j][i], sizeof(double));
	outfile.close();

	double time_output = (clock() - t_start_output) / (double)CLOCKS_PER_SEC;


	delete[] H;
	delete[] Ht;
	delete[] HU;
	delete[] H_ref;
	delete[] HUt;
	delete[] HV;
	delete[] HVt;
	delete[] Zdx;
	delete[] Zdy;

	//    %% Communicate time-to-compute

	double ops = nt * (15 + 2 + 11 + 30 + 30 + 1) * pow(nx, 2);
	double flops = ops / calculation_time;
	std::cout << "The allocation takes " << time_allocation << " s" << endl;
	std::cout << "The input takes " << time_input << " s" << endl;
	std::cout << "The calculation takes " << calculation_time << " s" << endl;
	std::cout << "The output takes " << time_output << " s" << endl;
	std::cout << "Average Performance " << flops / pow(10, 9) << " gfloops" << endl;

	return 0;
}

