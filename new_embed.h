#pragma once

#include <vector>
#include <complex>

// 添加命名空间
using namespace std;

int new_wm_embed(const wchar_t* path, const wchar_t* save_path, const int* wm, const int wm_size, const double start_time, const double end_time);

void YXju_embed(vector<vector<double>> Gp, vector<vector<double>> fp, vector<int> w1, double Delta, int M, int row, vector<vector<double>>& Gp_watermarked);

void YXfenjie513(vector<vector<double>> fp, int M, int Nmax, vector<complex<double>>& A_nm, vector<vector<int>>& zmlist);

void YXrec513(vector<complex<double>> A_nm_modified, vector<vector<int>> zmlist_selected, int M, vector<vector<complex<double>>>& result);