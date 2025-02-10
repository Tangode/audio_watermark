#pragma once

#include <vector>

using namespace std;

int new_wm_extract(const wchar_t* path, int* wm);

vector<double> YXju_extract(vector<vector<double>> Gp_w, vector<int> b, double Delta, int M);