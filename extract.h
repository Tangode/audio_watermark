#pragma once

#ifndef EXTRACT_H
#define EXTRACT_H

#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sndfile.h>
#include <opencv2/opencv.hpp>
#include <fftw3.h>
#include <string>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int wm_extract(const wchar_t* path, int* wm, int wm_size);

vector<double> PQIMfft_extract(MatrixXd Gp_w, vector<int> &b, int M, double DD, double EE, int pp);

double drzh(const vector<vector<int>>& ww, vector<vector<int>>& W1, int pp, int qq);

double psnrzh(const vector<double>& yw, const vector<double>& yo);

double getWsr(const vector<double>& yw, const vector<double>& yo);

vector<vector<int>> igeneral(vector<vector<int>> W1, int m, int n, int a, int b);

// std::vector<int> randperm(int length, unsigned int seed);

#endif