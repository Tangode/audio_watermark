#include <iostream>
#include <vector>
#include <sndfile.h>
#include <fmt/core.h>
#include <cmath>
#include "new_embed.h"
#include "util.h"
#include "dr_wav.h"
#include "embed.h"
#include "global.h"
#include "Log.h"

#define MATRIX_WIDTH_32 32;
#define MATRIX_WIDTH_16 16;

const double PI = 3.1416;

using namespace std;

int new_wm_embed(const wchar_t* path, const wchar_t* save_path, const int* wm, const int wm_size, const double start_time, const double end_time)
{
	Logger logger;
	// stage 1 - read audio file
	if (!canFormSquareMatrix(wm_size)) return 2;
	int is_mp3 = 0;
	int Fs = 0;
	int start = -1;
	int wavelength = 0;
	//vector<int> syn = { 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0 };
	vector<int> syn = { 1, 1, 1, 1, 1, 0, 0, 1 };
	const int Lsyn = 2 * syn.size();
	int channels;
	//int bitrate;
	vector<double> yo;
	vector<double> yo_origin;
	fs::path fname = path;
	fs::path s_path = save_path;
	string audio_name = ConvertUTF16ToMBCS(path);
	string save_name = ConvertUTF16ToMBCS(save_path);
	cout << "\n read file:" << audio_name << endl;
	if (isMp3File(fname, audio_name))
	{
		// is mp3 
		is_mp3 = 1;
		long Fs_long = 0;
		yo_origin = mp3ToDoubleVector(audio_name.c_str(), channels, Fs_long, wavelength);
		Fs = (int)Fs_long;
	}
	else {
		drwav wav;
		if (!drwav_init_file(&wav, audio_name.c_str(), nullptr)) {
			std::cerr << "Error opening WAV file." << std::endl;
			return 1;
		}
		yo_origin = readWav(wav, wavelength, channels, Fs);
	}

	if (Fs < 44100) {
		wstring resample_path = AddSuffixToPath(path, L"_resample");
		vector<double> out;
		convert_wav(yo_origin, Fs, channels, resample_path.c_str());
	}

	int max_wavelength = Fs * static_cast<int>(end_time - start_time) * channels;

	if ((end_time - start_time) < 25.0)
		// The difference between start_time and end_time is not less than 25 seconds
		return 4;

	bool split = yo_origin.size() > max_wavelength;

	if (yo.empty()) {
		// 如果音频长度大于25s 则根据start_time, end_time 来确定要嵌入水印的范围
		// getAudio20sFront(split, max_wavelength, Fs, channels, wavelength, yo_origin, yo);

		if (split) {
			start = static_cast<int>(start_time) * Fs * channels;
			cout << "start: " << start << endl;
			splitAudioByTime(max_wavelength, Fs, channels, wavelength, yo_origin, yo, start_time, end_time, start);
		}
		else {
			yo = yo_origin;
		}
	}

	vector<vector<double>> audio_data(wavelength, vector<double>(channels, 0.0));
	if (channels == 2)
	{
		for (int i = 0; i < yo.size(); i++)
		{
			if (i % 2 == 0) {
				audio_data[i / 2][0] = yo[i];
			}
			if (i % 2 == 1) {
				audio_data[i / 2][1] = yo[i];
			}
		}
	}
	else
	{
		for (int i = 0; i < yo.size(); i++)
		{
			audio_data[i][0] = yo[i];
		}
	}

	// stage 2 - preparation work for watermark
	std::vector<int> wm_vector(wm, wm + wm_size);
	// 输出结果
	std::cout << "wm_vector: ";
	for (int value : wm_vector) {
		std::cout << value << " ";
	}
	std::cout << std::endl;
	// enhance dimensions(2d)
	vector<vector<int>> ww_matrix = int_ott(wm_vector, 10);
	int rows = ww_matrix.size();
	int cols = ww_matrix[0].size();
	cout << "ww_rows:" << ww_matrix.size() << "\t ww_cols:" << ww_matrix[0].size() << endl;
	// image-scrambling
	vector<vector<int>> w = general(ww_matrix, rows, cols, 5, 6);
	// reduction dimension(1d)
	vector<int> w1 = int_tto(w, rows);

	// stage 3 - params init
	//double D		 = 0.8;
	//double Delta	 = 0.0008;
	int k1				= 5;
	int begin			= 4095;
	int block_length	= 65536;
	int i				= 1;
	//int left			= i * (Lsyn * k1 + block_length);
	int right			= wavelength - (Lsyn * k1 + block_length);
	int barkcode_length = k1 * syn.size();
	vector<int> t0;

	// stage 4 - start embed
	while (i * (Lsyn * k1 + block_length) < right)
	{
		cout << "start embed current batch No." << i << endl;
		int bb = begin + (i - 1) * (Lsyn * k1 + block_length);
		// embed syn code
		cout << "embed syn code at index: " << bb << endl;
		for (int mm = 0; mm < syn.size(); mm++)
		{
			vector<double> temp_vec_subsyn;
			int front = bb + (mm * k1) + 1;
			int back = bb + ((mm + 1) * k1);
			// calculate mean value
			double tempmean = calculateMean(audio_data, front, back, 0); // front = 4096 back = 4100
			int temp = static_cast<int>(round(tempmean / D));
			double tempmeanl;
			if (((temp) % 2 + 2) % 2 == syn[mm])
			{
				tempmeanl = temp * D + D / 2.0;
			}
			else {
				tempmeanl = temp * D - D / 2.0;
			}
			// update audio_data
			for (int j = front; j <= back; j++)
			{
				audio_data[j][0] += tempmeanl - tempmean;
				audio_data[j + barkcode_length][0] = audio_data[j][0];
			}
		}
		
		t0.push_back(begin + (i * Lsyn * k1) + ((i - 1) * block_length) + 1);
		// to embed watermark area
		vector<double> BLOCK1;
		for (int j = t0[i - 1]; j < t0[i - 1] + block_length; j++)
		{
			BLOCK1.push_back(audio_data[j][0]);
		}
		// wavelet
		vector<double> result = wavelet(BLOCK1, 1);
		result.resize(65536);
		BLOCK1.clear();
		vector<vector<double>> I = ott(result, 256);
		int n = I.size();  // rows
		int m = I[0].size();  // cols
		// convert to polar coordinates
		int M = 4 * n;
		vector<vector<double>> fp(M, vector<double>(M, 0.0));
		vector<vector<double>> Gp(M, vector<double>(M, 0.0));
		vector<vector<double>> rr(M, vector<double>(M, 0.0));
		vector<vector<double>> oo(M, vector<double>(M, 0.0));
		vector<vector<int>>	   vv(M, vector<int>(M, 0));
		vector<vector<int>>	   uu(M, vector<int>(M, 0));

		for (int u = 0; u < M; u++) {
			for (int v = 0; v < M; v++) {
				double rrr = static_cast<double> (u) / M;
				double ooo = (2 * PI * v) / M;
				int kk = ceil(rrr * (n / 2) * sin(ooo));
				int ll = ceil(rrr * (n / 2) * cos(ooo));
				fp[u][v] = I[(-1) * kk + (n / 2)][ll + (n / 2) - 1];
				Gp[u][v] = fp[u][v] * sqrt(static_cast<double> (u) / (2 * M));
			}
		}

		for (int k = 0; k < M; k++) {
			for (int j = 0; j < M; j++) {
				vv[k][j] = j + 1;
				uu[k][j] = k + 1;
			}
		}

		for (int k = 0; k < M; k++) {
			for (int j = 0; j < M; j++) {
				rr[k][j] = static_cast<double>(uu[k][j] - 1) / M;
				oo[k][j] = static_cast<double>(2 * PI * (vv[k][j] - 1)) / M;
			}
		}

		vector<vector<double>> Gp_watermarked(M, vector<double>(M, 0.0));

		YXju_embed(Gp, fp, w1, Delta, M, rows, Gp_watermarked);

		vector<vector<double>> I_incircle(n, vector<double>(m, 0.0));
		cout << "\n embed watermark" << endl;

		for (int u = 0; u < M; u++) {
			for (int v = 0; v < M; v++) {
				double r = static_cast<double>(u) / M;
				double o = (2 * PI * v) / M;
				int kkk = static_cast<int>(std::ceil(r * (n / 2) * std::sin(o)));
				int lll = static_cast<int>(std::ceil(r * (n / 2) * std::cos(o)));
				I_incircle[-kkk + (n / 2)][lll + (n / 2) - 1] = Gp_watermarked[u][v];
			}
		}

		cout << "I_incircle's rows:" << I_incircle.size() << "I_incircle's cols: " << I_incircle[0].size() << endl;

		for (int ii = 0; ii < n; ii++) {
			for (int jj = 0; jj < m; jj++) {
				if (I_incircle[ii][jj] == 0) {
					I_incircle[ii][jj] = I[ii][jj];
				}
			}
		}

		cout << "\n go to tto;" << endl;
		vector<double> BLOCK2 = tto(I_incircle, 256);
		vector<double> new_BLOCK2(BLOCK2.size(), 0.0);
		BLOCK2.insert(BLOCK2.end(), new_BLOCK2.begin(), new_BLOCK2.end());
		cout << "\n go to inverseWavelet;" << endl;
		vector<double> signal = inverseWavelet(BLOCK2, 1);

		for (int j = t0[i - 1]; j < t0[i - 1] + block_length; j++) {
			audio_data[j][0] = signal[j - t0[i - 1]];
		}

		double sub_progress = double(i * (Lsyn * k1 + block_length)) / (wavelength - (Lsyn * k1 + block_length)) * 100;
		g_audio_wm_progress = sub_progress;
		cout << "\n total progress: " << g_audio_wm_progress << endl;
		cout << "==============================================================" << endl;

		i++;
		//left = i * (Lsyn * k1 + block_length);
	}
	//trans audio_data to yo
	printIntSignal(t0, 0);
	yo.clear();
	if (channels == 2) {
		for (int j = 0; j < audio_data.size(); j++)
		{
			yo.push_back(audio_data[j][0]);
			yo.push_back(audio_data[j][1]);
		}
	}
	else {
		for (int j = 0; j < audio_data.size(); j++)
		{
			yo.push_back(audio_data[j][0]);
		}
	}

	if (split)
	{
		for (size_t i = 0; i < yo.size(); ++i) {
			yo_origin[start + i] = yo[i];
		}
	}
	else {
		yo_origin = yo;
	}
	// stage 5 - save audio
	fs::path full_save_path = s_path;
	const wchar_t* wchar_save_path = full_save_path.c_str();
	if (is_mp3) {
		save_audio_drmp3(wchar_save_path, yo_origin, Fs, channels);
	}
	else {
		save_audio_drwav(wchar_save_path, yo_origin, Fs, channels);
	}

	return 0;
}

void YXju_embed(vector<vector<double>> Gp, vector<vector<double>> fp, vector<int> w1, double Delta, int M, int row, vector<vector<double>> &Gp_watermarked)
{
	int						L = w1.size();
	int						Nmax = 11;
	int						size = 2 * Nmax + 1;
	int						total_elements = (Nmax + 1) * size;
	vector<vector<int>>		zmlist(total_elements, vector<int>(2, 0));
	vector<complex<double>> A_nm(total_elements, (0, 0));
	vector<vector<int>>		zmlist_selected(total_elements, vector<int>(2, 0));
	vector<complex<double>> A_nm_selected;
	vector<complex<double>> A_nm_modified;

	YXfenjie513(Gp, M, Nmax, A_nm, zmlist);

	vector<int> index_n(zmlist.size(), 0);
	vector<int> index_m(zmlist.size(), 0);
	for (int i = 0; i < zmlist.size(); i++) {
		index_n[i] = zmlist[i][0];
		index_m[i] = zmlist[i][1];
	}
	// choose suitble embed range
	vector<int> index_suitable;
	for (int i = 0; i < zmlist.size(); i++) {
		// 10*10 embed range
		if (((index_m[i] >= 0) && (index_m[i] <= 18) && (index_n[i] >= 0) && (index_n[i] <= 17))) {
			index_suitable.push_back(i);
		}
	}
	index_n.clear();
	index_m.clear();
	vector<int> index_selected_embed(index_suitable.begin(), index_suitable.begin() + L);
	index_suitable.clear();
	zmlist_selected.resize(L, vector<int>(2, 0));
	A_nm_selected.resize(L, (0, 0));

	for (int i = 0; i < L; i++) {
		zmlist_selected[i] = zmlist[index_selected_embed[i]];
		A_nm_selected[i]   = A_nm[index_selected_embed[i]];
	}

	index_selected_embed.clear();
	A_nm_modified = A_nm_selected;
	vector<double> II(L, 0.0);

	for (int i = 0; i < L; i++) {
		complex<double> A_selected = A_nm_modified[i];
		II[i] = floor(abs(A_selected) / Delta);
		int temp = fmod(abs(II[i]), 2.0);
		if (fmod(abs(II[i]), 2.0) == w1[i])
		{
			II[i] = abs(II[i]) * Delta;
		}
		else {
			II[i] = abs(II[i]) * Delta + Delta;
		}
		A_nm_modified[i] = II[i] / abs(A_selected) * A_selected;
	}

	II.clear();
	
	vector<vector<complex<double>>> Gp_rec_beforemodify;
	YXrec513(A_nm_selected, zmlist_selected, M, Gp_rec_beforemodify);
	vector<vector<complex<double>>> Gp_rec;
	YXrec513(A_nm_modified, zmlist_selected, M, Gp_rec);

	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			Gp_watermarked[i][j] = fp[i][j] - std::real(Gp_rec_beforemodify[i][j]) + std::real(Gp_rec[i][j]);
		}
	}
}

void YXfenjie513(vector<vector<double>> fp, int M, int Nmax, vector<complex<double>> &A_nm, vector<vector<int>> &zmlist)
{
	vector<double>					r(M);
	vector<double>					o(M);
	vector<vector<double>>			Tn(Nmax + 1, vector<double>(M));
	// Compute rrr and ooo
	for (int u = 0; u < M; ++u) {
		r[u] = static_cast<double>(u) / M;
	}

	for (int v = 0; v < M; ++v) {
		o[v] = static_cast<double>(2 * PI * v) / M;
	}

	// Compute radial functions
	for (int k = 0; k < Nmax + 1; ++k) 
	{
		for (int u = 0; u < M; ++u) 
		{
			for (int v = 0; v < M; ++v) {
				if (k == 0) {
					Tn[k][u] = sqrt(r[u]);
				}
				else if ((k % 2 + 2) % 2 == 0) {
					Tn[k][u] = sqrt(2.0 * r[u]) * cos(PI * k * r[u]);
				}
				else {
					Tn[k][u] = sqrt(2.0 * r[u]) * sin(PI * (k + 1) * r[u]);
				}
			}
		}
	}
	int c = 0;
	for (int k = 0; k < Nmax + 1; ++k) {
		for (int m = 0; m < (2 * Nmax + 1); ++m) {
			complex<double> temp_sum = 0.0;
			for (int u = 0; u < M; ++u) {
				double sub_tn = Tn[k][u];
				for (int v = 0; v < M; ++v) {
					double item = -(m - Nmax) * o[v];
					double cos_img_part = cos(item);
					double sin_img_part = sin(item);
					double front_part = fp[u][v] * sub_tn;
					complex<double> temp = front_part * complex<double>(cos_img_part, sin_img_part);
					temp_sum += temp;
				}
			}
			zmlist[c][0] = k;
			zmlist[c][1] = m - Nmax;
			A_nm[c] = (1.0 / (M * M)) * temp_sum;
			c++;
		}
	}
}

void YXrec513(vector<complex<double>> A_nmm, vector<vector<int>> zmlistt, int M, vector<vector<complex<double>>> &result)
{
	cout << "YXrec513" << endl;
	// Initialize the output matrix gg
	result.resize(M, vector<complex<double>>(M, 0.0));
	// Create rrr and ooo matrices
	vector<double> r(M);
	vector<double> o(M);

	for (int u = 0; u < M; ++u) {
		r[u] = static_cast<double>(u) / M;
	}

	for (int v = 0; v < M; ++v) {
		o[v] = (2 * PI * v) / M;
	}
	// Process each element in A_nmm and zmlistt
	for (size_t kk = 0; kk < A_nmm.size(); ++kk) {
		int nj = zmlistt[kk][0];
		int mc = zmlistt[kk][1];
		vector<double> Ta(M, 0.0);
		//vector<double> Ta(M);
		if (nj == 0) {
			for (int u = 0; u < M; ++u) {
				for (int v = 0; v < M; ++v) {
					Ta[u] = sqrt(1.0 / r[u]);
				}
			}
		}
		else if ((nj % 2 + 2) % 2 == 0) {
			for (int u = 0; u < M; ++u) {
				for (int v = 0; v < M; ++v) {
					Ta[u] = sqrt(2.0 / r[u]) * cos(PI * nj * r[u]);
				}
			}
		}
		else {
			for (int u = 0; u < M; ++u) {
				for (int v = 0; v < M; ++v) {
					Ta[u] = sqrt(2.0 / r[u]) * sin(PI * (nj + 1) * r[u]);
				}
			}
		}

		if (mc == 0) {
			for (int u = 0; u < M; ++u) {
				double Ta_item = Ta[u];
				for (int v = 0; v < M; ++v) {
					double cos_img_part = cos(mc * o[v]);
					double sin_img_part = sin(mc * o[v]);
					complex<double> front = A_nmm[kk] * Ta_item;
					result[u][v] += front * complex<double>(cos_img_part, sin_img_part);
					
				}
			}
		}
		else {
			for (int u = 0; u < M; ++u) {
				double Ta_item = Ta[u];
				complex<double> front_sub = A_nmm[kk] * Ta_item;
				complex<double> end_sub = conj(A_nmm[kk]) * Ta_item;
				for (int v = 0; v < M; ++v) {
					double cos_img_part = cos(mc * o[v]);
					double sin_img_part = sin(mc * o[v]);
					complex<double> front = front_sub * complex<double>(cos_img_part, sin_img_part);
					complex<double> end = end_sub * complex<double>(cos_img_part, -sin_img_part);
					result[u][v] += front + end;
				}
			}
		}
	}
}