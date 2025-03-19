#pragma once

#include <string>
#include <vector>
#include <complex>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <Eigen/Dense>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;
using namespace Eigen;


struct WAVHeader {
    char riff[4];            
    uint32_t fileSize;       
    char wave[4];            
    char fmt[4];             
    uint32_t fmtSize;        
    uint16_t audioFormat;    
    uint16_t numChannels;    
    uint32_t sampleRate;     
    uint32_t byteRate;       
    uint16_t blockAlign;     
    uint16_t bitsPerSample;  
    char data[4];            
    uint32_t dataSize;       
};

vector<double> tto(std::vector<std::vector<double>>& I, int N);

vector<int> int_tto(std::vector<std::vector<int>>& I, int N);

std::vector<std::vector<double>> ott(const std::vector<double>& I, int N);

std::vector<std::vector<int>> int_ott(const std::vector<int>& I, int N);

//void haarWavelet(vector<double> &signal, int numLevels);
//
//void haarWaveletTransform(vector<double> &signal);
//
//void inverseHaarWavelet(vector<double> &signal, int numLevels);
//
//void inverseHaarWaveletTransform(vector<double> &signal);

void fft1d(vector<complex<double>>& data, bool invert);

void fft2(Mat& input, Mat& output, bool invert, bool normalize);

void zhishujufenjie62(MatrixXd Gp, int M, int Nmax, vector<complex<double>> &A_nm, vector<vector<int>> &zmlist);

//void printMessage(vector<string> &msg);

vector<int16_t> floatToPCM16(std::vector<double>& input, float scaleFactor);

double getGlobalProgress();

void embedWatermark();

void save_audio_drwav(const wchar_t* outputFilename, vector<double>& yo, int sampleRate, int channels);

void save_audio_drmp3(const wchar_t* outputFilename, vector<double>& yo, int sampleRate, int channels);

vector<double> mp3ToDoubleVector(const char* mp3File, int& channels, long& rate, int& wavelength);

bool isMp3File(fs::path& path, string fpath);

string toUTF8(const wstring& wstr);

wstring toWideString(const std::string& str);

void modifyAndCombinePath(const char* path, const char* save_path, char* output);

string ConvertUTF16ToMBCS(const wchar_t* utf16Str);

bool transResampleReadWAV(const char* inputPath, vector<float>& buffer, int& sampleRate, int& channels);

bool transResampleWriteWAV(const char* outputPath, const vector<float>& buffer, double sampleRate, int channels);

void resample(const float* inputBuffer, size_t inputSize, float* outputBuffer, size_t& outputSize, double inputRate, double outputRate, int channels);

void convert_wav(vector<double>& inputBuffer, int& inputRate, int channels, const wchar_t* outputPath);

wstring AddSuffixToPath(const wchar_t* originalPath, const wstring& suffix);

bool allEqual(const vector<double>& vec1, const vector<double>& vec2);