#include "extract.h"
#include "util.h"
#include "embed.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sndfile.h>
#include <opencv2/opencv.hpp>
#include <fftw3.h>
#include <string>
#include <complex>
#include <cstdint>
#include <numeric>
#include <random>
#include "new_embed.h"
#include "new_extract.h"

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int new_wm_extract(const wchar_t* path, int* wm)
{
	// stage 1 - read file
	cout << "stage 1 - read file" << endl;
	fs::path fname = path;
	vector<double> yw;
	vector<double> yw_origin;
	int wavelength;
	int Fs;
	int numChannels;
	string extract_path = ConvertUTF16ToMBCS(path);
	if (isMp3File(fname, extract_path))
	{
		// is mp3 
		long Fs_long = 0;
		yw = mp3ToDoubleVector(extract_path.c_str(), numChannels, Fs_long, wavelength);
		Fs = (int)Fs_long;
	}
	else {
		drwav wav;
		if (!drwav_init_file(&wav, extract_path.c_str(), nullptr)) {
			std::cerr << "Error opening WAV file." << std::endl;
			return 1;
		}
		yw = readWav(wav, wavelength, numChannels, Fs);
	}

	vector<vector<double>> audio_data;

	if (numChannels == 1) {
		for (int i = 0; i < yw.size(); i++)
		{
			audio_data.push_back({ yw[i], 0 });
		}
	}
	else {
		for (int i = 0; i < yw.size(); i++)
		{
			if (i % numChannels == 0)
			{
				audio_data.push_back({ yw[i], 0 });
			}
		}
	}

	vector<int> syn = { 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0 };
	vector<int> synw = syn;
	vector<int> temp1 = syn;
	int Lsyn = syn.size();
	int pp = 10;
	int qq = 10;
	vector<int> b;
	vector<vector<int>> W1(pp, vector<int>(qq, 0));
	// stage 2 - params init
	cout << "stage 2 - params init" << endl;
	int k1 = 5;
	double PI = 3.1416;
	double D = 1;
	//double Delta = 0.0008;
	//int blocklength = pp * qq * 8 * 8; // wm size 32*32
	int blocklength = 65536;
	int k = 0;
	int i = 1;
	int rela = 1;
	vector<int> t0;
	cout << "wavelength: " << wavelength << endl;
	// stage 3 - position syn code
	cout << "stage 3 - position syn code" << endl;
	while (k + (i * Lsyn * k1) < wavelength)
	{
		while (rela != 0 && (k + Lsyn * k1) < wavelength)
		{
			for (int mm = 0; mm < Lsyn; mm++) {
				double tempmean = 0.0;
				int front = k + mm * k1;
				int back = k + (mm + 1) * k1 - 1;
				tempmean = calculateMean(audio_data, front, back, 0);
				int temp = floor(tempmean / D);
				synw[mm] = (temp % 2 + 2) % 2;
				temp1[mm] = syn[mm] ^ synw[mm];
			}
			rela = accumulate(temp1.begin(), temp1.end(), 0);
			k++;
		}
		t0.push_back(k - 1);
		rela = 1;
		k = k + (Lsyn * k1) + blocklength - 1;
		i++;
	}
	printIntSignal(t0, 0);
	cout << "\nstage 4 - watermark extract" << endl;
	vector<double> bt_temp;
	vector<vector<double>> bt;
	vector<int> t1;
	for (int ii = 0; ii < i - 1; ii++)
	{
		cout << "extract watermark current batch No." << ii + 1 << endl;
		int t1_item = t0[ii] + Lsyn * k1 + 1;
		t1.push_back(t1_item);
		if (t1[ii] + blocklength - 1 > wavelength) {
			//W1 = vector<vector<int>>(pp, vector<int>(qq, 1));
			W1[pp - 1][qq - 1] = 1;
		}
		else {
			vector<double> BLOCK(blocklength);

			for (int idx = 0; idx < blocklength; idx++) {
				BLOCK[idx] = audio_data[t1[ii] + idx][0];
			}

			vector<double> I_w = wavelet(BLOCK, 1);
			vector<vector<double>> I_w1 = ott(I_w, 256);
			int n = I_w1.size();
			int m = I_w1[0].size();
			if ((n % 2 + 2) % 2 != 0) {
				n = n - 1;
			}
			int M = 4 * n;
			vector<vector<double>> Gp_w(M, vector<double>(M, 0.0));
			for (int u = 0; u < M; u++) {
				for (int v = 0; v < M; v++) {
					double rrr = static_cast<double>(u) / M;
					double ooo = (2 * PI * v) / M;
					int kk = ceil(rrr * (n / 2.0) * sin(ooo));
					int ll = ceil(rrr * (n / 2.0) * cos(ooo));

					int row = -kk + (n / 2);
					int col = ll + (n / 2) - 1;
					double f = I_w1[row][col];
					double e = sqrt(static_cast<double>(u) / (2 * M));
					Gp_w[u][v] = f * e;
				}
			}

			bt_temp = YXju_extract(Gp_w, b, Delta, M);
		}
			// extract watermark

			cout << "No." << ii + 1 << " bt_temp" << endl;
			printSignal(bt_temp, 0);
			cout << endl;
			bt.push_back(bt_temp);
		
	}
	printIntSignal(t1, 0);
	cout << "stage 5 - watermark reconstruction" << endl;
	vector<int> r(100);

	for (int iii = 0; iii < 100; iii++) {
		double sum_bt = 0.0;

		for (int j = 0; j < bt.size(); j++) {
			sum_bt += bt[j][iii];
		}
		//cout << "iii = " << iii << ";\tbt.size: " << bt.size() << ";\tsum_bt: " << sum_bt << ";\tres = " << sum_bt / (bt.size()) << ";\tround_res = " << round(sum_bt / (bt.size())) 
		//	<< "; final_res = " << static_cast<int>(round(sum_bt / (bt.size() - 2))) << endl;

		r[iii] = static_cast<int>(round(sum_bt / (bt.size() - 2)));
	}

	for (size_t i = 0; i < r.size(); ++i) {
		wm[i] = r[i];
	}
	cout << "stage 6 - watermark igeneral" << endl;
	W1 = int_ott(r, 10);
	vector<vector<int>> W2 = igeneral(W1, pp, qq, 5, 6);
	cout << "\n W1:" << endl;
	printIntMat(W1);
	cout << "\n W2:" << endl;
	printIntMat(W2);
	vector<int> wm_vec = int_tto(W2, 10);

	if (wm == nullptr || wm_vec.size() > 100) {
		return 3; // 返回错误代码
	}

	for (size_t i = 0; i < wm_vec.size(); ++i) {
		wm[i] = static_cast<char>(wm_vec[i]);
	}

	wm[wm_vec.size()] = '\0'; // 确保字符串以 null 结尾

	return 0;
}

vector<double> YXju_extract(vector<vector<double>> Gp_w, vector<int> b, double Delta, int M)
{
	int						L = 100;
	int						Nmax = 11;
	int						size = 2 * Nmax + 1;
	int						total_elements = (Nmax + 1) * size;
	vector<vector<int>>		zmlist(total_elements, vector<int>(2, 0));
	vector<complex<double>> A_nm(total_elements, (0, 0));
	vector<vector<int>>		zmlist_selected(total_elements, vector<int>(2, 0));
	vector<complex<double>> A_nm_selected;
	vector<complex<double>> A_nm_modified;

	YXfenjie513(Gp_w, M, Nmax, A_nm, zmlist);

	vector<int> index_suitable;
	vector<complex<double>> A_nm_selected_extract(L, 0.0);
	vector<double> b_extract(L, 0.0);

	for (int i = 0; i < zmlist.size(); i++) {
		int index_n_temp = zmlist[i][0];
		int index_m_temp = zmlist[i][1];

		// 10*10 embed range
		if ((index_m_temp >= 0) && (index_m_temp <= 18) && (index_n_temp >= 0) && (index_n_temp <= 17)) {
			index_suitable.push_back(i);
		}
	}

	for (int i = 0; i < L; i++)
	{
		A_nm_selected_extract[i] = A_nm[index_suitable[i]];
	}

	for (int i = 0; i < L; i++)
	{
		double temp = round(abs(A_nm_selected_extract[i]) / Delta);

		if (fmod(temp, 2.0) == 1)
		{
			b_extract[i] = 1;
		}
		else {
			b_extract[i] = 0;
		}
	}

	return b_extract;
}