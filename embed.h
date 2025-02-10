#pragma once

#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sndfile.h>
#include <opencv2/opencv.hpp>
#include <fftw3.h>
#include <string>
#include <complex>
#include "global.h"
#include "dr_wav.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

vector<vector<int>> general(const vector<vector<int>>& w, int m, int n, int a, int b);

std::vector<std::vector<int>> matToVector(const cv::Mat& ww, int rows, int cols);

void printSignal(vector<double> &signal, int length);

void PQIMfft_embed(vector<vector<double>> &Gp, vector<vector<double>> &fp, vector<int> &w1, int M, double DD, double EE, vector<vector<double>> &oo, vector<vector<double>> &rr, vector<vector<double>> &Gp_watermarked);

void saveAudio(const char* outputFilename, std::vector<double>& yo, int sampleRate, int channels);
// 对于复数的sign函数，返回其实部和虚部的符号组成的复数  
template<typename T>

std::complex<T> complex_sign(const std::complex<T> &z);

void zhishujurec622(vector<complex<double>> A_nm, vector<vector<int>> &zmlist_selected, int M, vector<vector<double>> oo, vector<vector<double>> rr, vector<vector<complex<double>>> &result);

int wm_embed(const wchar_t* path, const wchar_t* save_path, const int* wm, const int wm_size, const double start_time = 0, const double end_time = 25);

void printIntSignal(vector<int> &signal, int length);

void printMat(vector<vector<double>> mat);

double calculateMean(std::vector<std::vector<double>> &yo, int start, int end, int column);

vector<double> wavelet(vector<double> BLOCK1, int level);

vector<double> inverseWavelet(vector<double> signal, int level);

void printIntMat(std::vector<std::vector<int>> mat);

//void wmLog(char* txt);

//void save_audio_drmp3(const char* outputFilename, vector<double>& yo, int sampleRate, int channels);

vector<double> readWav(drwav& wav, int& wavelength, int& channels, int& Fs);

bool canFormSquareMatrix(const int wm_size);

void getAudio20sFront(bool split, int max_wavelength, int Fs, int channels, int& wavelength, vector<double> yo_origin, vector<double>& yo);

void splitAudioByTime(int max_wavelength, int Fs, int channels, int& wavelength, vector<double> yo_origin, vector<double>& yo, const double start_time, const double end_time, int start);
