#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <sndfile.h>
#include <fmt/core.h>
#include <windows.h>
#include <fstream>
#include <cmath>
#include "global.h"
#include "dr_wav.h"
#include "embed.h"
#include "util.h"
#include "wavelet.h"
//#include "miniaudio.h"
#include "dr_mp3.h"
#include <numeric>
#include <sstream>
#include <cstring>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

const int						k1 = 5;
const double					PI = 3.1416;
int                             G_L = 0;
const int                       G_NMAX_32 = 40;
const int                       G_SIZE_32 = 2 * G_NMAX_32 + 1;
const int                       G_TOTAL_ELEMENTS_32 = G_SIZE_32 * G_SIZE_32;
vector<vector<int>>             G_ZMLIST_32(G_TOTAL_ELEMENTS_32, vector<int>(2, 0));
vector<complex<double>>         G_A_NM_32(G_TOTAL_ELEMENTS_32, (0, 0));
vector<vector<int>>             G_ZMLIST_SELECTED_32(G_TOTAL_ELEMENTS_32, vector<int>(2, 0));
const int                       G_NMAX_10 = 20;
const int                       G_SIZE_10 = 2 * G_NMAX_10 + 1;
const int                       G_TOTAL_ELEMENTS_10 = G_SIZE_10 * G_SIZE_10;
vector<vector<int>>             G_ZMLIST_10(G_TOTAL_ELEMENTS_10, vector<int>(2, 0));
vector<complex<double>>         G_A_NM_10(G_TOTAL_ELEMENTS_10, (0, 0));
vector<vector<int>>             G_ZMLIST_SELECTED_10(G_TOTAL_ELEMENTS_10, vector<int>(2, 0));
vector<complex<double>>         G_A_NM_SELECTED;
vector<complex<double>>         G_A_NM_MODIFIED;
vector<vector<complex<double>>> G_GP_REC_BEFOREMODIFY;
vector<vector<complex<double>>> G_GP_REC;

int wm_embed(const wchar_t* path, const wchar_t* save_path, const int* wm, const int wm_size, const double start_time, const double end_time) {
	if (!canFormSquareMatrix(wm_size)) return 2;
	int is_mp3 = 0;
	int Fs = 0;
	int wavelength = 0;
	double D = 0.8; // init 0.2
	double DD = 0.0012; // init DD 0.0012
	double EE = 0.00002; // init EE 0.00002
	int begin = 4095;
	int block_length = 0;
	vector<int> syn = {1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0};
	const int Lsyn = syn.size();
	fs::path fname = path;
	fs::path s_path = save_path;
	int channels;
	vector<double> yo;
	vector<double> yo_origin;
	int start = 0;
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
	// stage 1 start
	//const int frame_size = max_wavelength; // 每帧大小
	//const int hop_size = Fs; // 帧移
	//size_t num_frames = (yo_origin.size() - frame_size) / hop_size + 1;
	//std::vector<double> energy(num_frames);

	//// Calculate short-term energy
	//for (size_t i = 0; i < num_frames; ++i) {
	//	double frame_energy = 0.0;
	//	for (int j = 0; j < frame_size; ++j) {
	//		frame_energy += yo_origin[i * hop_size + j] * yo_origin[i * hop_size + j];
	//	}
	//	energy[i] = frame_energy;
	//}

	//// Calculate the mean and standard deviation of energy
	//double mean_energy = std::accumulate(energy.begin(), energy.end(), 0.0) / num_frames;
	//double sq_sum = std::inner_product(energy.begin(), energy.end(), energy.begin(), 0.0);
	//double stddev_energy = std::sqrt(sq_sum / num_frames - mean_energy * mean_energy);

	//// Set a threshold, such as adding twice the standard deviation to the mean
	//double threshold = mean_energy + 2 * stddev_energy;

	//// Detect noisy areas
	//for (size_t i = 0; i < num_frames; ++i) {
	//	if (energy[i] > threshold) {
	//		start = i * hop_size;
	//		std::cout << "Noisy frame detected at frame index: " << i << ", yo_origin[" << i * hop_size << "], sec: " << static_cast<double>(i * hop_size) / (Fs * 2) << "s,Energy: " << energy[i] << std::endl;

	//		// 将噪杂的帧添加到yo中
	//		for (int j = 0; j < max_wavelength; ++j) {
	//			yo.push_back(yo_origin[start + j]);
	//		}

	//		wavelength = max_wavelength / channels;
	//		break;
	//	}
	//}

	// stage 1 end
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

	vector<vector<double>> audio_data(yo.size() / channels, vector<double>(channels, 0.0));
	cout << "yo.size():" << yo.size() << endl;
	cout << "audio_data.size():" << audio_data.size() << endl;
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
	std::vector<int> wm_vector(wm, wm + wm_size);
	//vector<vector<int>> ww_matrix = matToVector(ww, rows, cols);
	vector<vector<int>> ww_matrix = int_ott(wm_vector, 10);
	int rows = ww_matrix.size();
	int cols = ww_matrix[0].size();
	cout << "ww_rows:" << ww_matrix.size() << "\t ww_cols:" << ww_matrix[0].size() << endl;
	// 图像置乱
	vector<vector<int>> w = general(ww_matrix, rows, cols, 5, 6);
	// 降维
	vector<int> w1 = int_tto(w, rows);
	//block_length = rows * cols * 8 * 8;
	block_length = 65536;
	int i = 1;
	int left = i * (Lsyn * k1 + block_length);
	int right = wavelength - (Lsyn * k1 + block_length) * 2;
	vector<int> t0;

	while (i * (Lsyn * k1 + block_length) < wavelength - ((Lsyn * k1 + block_length) * 2))
	{
		int bb = begin + (i - 1) * (Lsyn * k1 + block_length);

		cout << "\n embed barker code form " << bb << " to " << bb + Lsyn * k1 << endl;

		for (int mm = 0; mm < Lsyn; mm++)
		{
			vector<double> temp_vec_subsyn;

			int front = bb + (mm * k1);

			int back = bb + ((mm + 1) * k1);
			//计算指定范围均值
			double tempmean = calculateMean(audio_data, front + 1, back, 0); // front = 4096 back = 4100
			
			int temp = static_cast<int>(floor(tempmean / D));

			double tempmeanl;

			if ((temp % 2 + 2) % 2 == syn[mm])
			{
				tempmeanl = temp * D + D / 2.0;
			}
			else {
				tempmeanl = temp * D - D / 2.0;
			}
			// 更新指定范围的yo
			for (int j = front; j <= back; j++)
			{
				audio_data[j][0] += tempmeanl - tempmean;
			}
		}

		t0.push_back(begin + (i * Lsyn * k1) + ((i - 1) * block_length) + 1);

		vector<double> BLOCK1;
		for (int j = t0[i - 1]; j < t0[i - 1] + block_length; j++)
		{
			BLOCK1.push_back(audio_data[j][0]);
		}
		vector<double> result = wavelet(BLOCK1, 1);

		result.resize(65536);

		BLOCK1.clear();

		vector<vector<double>> I = ott(result, 256);

		int n = I.size();  // 行数

		int m = I[0].size();  // 列数

		// 极坐标转换相关参数
		int M = 4 * n;

		vector<vector<double>> fp(M, vector<double>(M, 0.0));
		vector<vector<double>> Gp(M, vector<double>(M, 0.0));
		vector<vector<double>> rr(M, vector<double>(M, 0.0));
		vector<vector<double>> oo(M, vector<double>(M, 0.0));

		// 声明和初始化 vv 和 uu 矩阵
		vector<vector<int>> vv(M, vector<int>(M, 0));
		vector<vector<int>> uu(M, vector<int>(M, 0));

		// 填充 fp 和 Gp 数组
		for (int u = 0; u < M; u++) {
			for (int v = 0; v < M; v++) {
				double rrr = static_cast<double> (u) / M;
				double ooo = (2 * PI * v) / M;
				int kk = ceil(rrr * (n / 2) * sin(ooo));
				int ll = ceil(rrr * (n / 2) * cos(ooo));
				fp[u][v] = I[(-1) * kk + (n / 2)][ll + (n / 2) - 1];
				Gp[u][v] = fp[u][v] * std::sqrt(static_cast<double> (u) / (2 * M));
			}
		}

		// 填充 vv 和 uu 矩阵
		for (int k = 0; k < M; k++) {
			for (int j = 0; j < M; j++) {
				vv[k][j] = j + 1;
				uu[k][j] = k + 1;
			}
		}
		
		// 计算 rr 和 oo 矩阵
		for (int k = 0; k < M; k++) {
			for (int j = 0; j < M; j++) {
				rr[k][j] = static_cast<double>(uu[k][j] - 1) / M;
				oo[k][j] = static_cast<double>(2 * PI * (vv[k][j] - 1)) / M;
			}
		}

		vector<vector<double>> Gp_watermarked(M, vector<double>(M, 0.0));

		PQIMfft_embed(Gp, fp, w1, M, DD, EE, oo, rr, Gp_watermarked);

		cout << "初始化 I_incircle 为全零矩阵" << endl;

		vector<vector<double>> I_incircle(n, vector<double>(m, 0.0));
		// 嵌入水印
		cout << "\n 嵌入水印" << endl;

		for (int u = 0; u < M; u++) {
			for (int v = 0; v < M; v++) {
				double r = static_cast<double>(u) / M;
				double o = (2 * PI * v) / M;
				int kkk = static_cast<int>(std::ceil(r * (n / 2) * std::sin(o)));
				int lll = static_cast<int>(std::ceil(r * (n / 2) * std::cos(o)));
				I_incircle[-kkk + (n / 2)][lll + (n / 2) - 1] = Gp_watermarked[u][v];
			}
		}
		// 重构水印
		cout << "重构水印" << endl;
		// vector<vector<double>> I_watermarked = I_incircle;
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

		//cout << "\n NO." << i << " , form " << t0[i - 1] << " to " << t0[i - 1] + block_length - 1 << endl;

		for (int j = t0[i - 1]; j < t0[i - 1] + block_length; j++) {
			audio_data[j][0] = signal[j - t0[i - 1]];
		}
		
		double sub_progress = double(i * (Lsyn * k1 + block_length)) / (wavelength - (Lsyn * k1 + block_length) * 2) * 100;

		g_audio_wm_progress = sub_progress;

		cout << "\n 总完成进度:" << sub_progress << endl;
		cout << "==============================================================" << endl;

		i++;
		left = i * (Lsyn * k1 + block_length);
	}
	cout << "t0:" << endl;
	printIntSignal(t0, 0);
	//trans audio_data to yo
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
	cout << "yo.size: " << yo.size() << endl;
	printSignal(yo, 10);
	cout << "start: " << start << endl;
	if (split)
	{
		for (size_t i = 0; i < yo.size(); ++i) {
			yo_origin[start + i] = yo[i];
		}
	}
	else 
	{
		yo_origin = yo;
	}
	// save audio
	const wchar_t* wchar_save_path = s_path.c_str(); // 使用完整的保存路径
	if (is_mp3) {
		save_audio_drmp3(wchar_save_path, yo_origin, Fs, channels);
	}
	else {
		save_audio_drwav(wchar_save_path, yo_origin, Fs, channels);
	}
	//fs::remove(fname);
	return 0;
}

void saveAudio(const char* outputFilename, vector<double> &yo, int sampleRate, int channels) {
	SF_INFO sfinfo{};
	sfinfo.samplerate = sampleRate;   // 设置采样率
	sfinfo.channels = channels;         // 设置声道数
	sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;  // 设置文件格式
	// todo save audio
	SNDFILE* outfile = sf_open(outputFilename, SFM_WRITE, &sfinfo);

	sf_count_t count = sf_write_double(outfile, yo.data(), yo.size());
	cout << "\n保存地址:" << outputFilename << endl;
	if (count != yo.size()) {
		std::cerr << "Error writing frames to file" << std::endl;
		sf_close(outfile);
		return;
	}
	// 关闭文件
	sf_close(outfile);
	std::cout << "\nFile saved successfully." << std::endl;
}

vector<double> inverseWavelet(vector<double> swt_output, int level) {
	vector<double> iswt_output;

	iswt(swt_output, level, "haar", iswt_output);

	return iswt_output;
}

void PQIMfft_embed(vector<vector<double>> &Gp, vector<vector<double>>& fp,
					vector<int> &w1, int M, double DD, double EE, vector<vector<double>> &oo,
					vector<vector<double>> &rr, vector<vector<double>> &Gp_watermarked)
{
	G_L = w1.size();

	zhishujufenjie62(Gp, M, G_NMAX_10, G_A_NM_10, G_ZMLIST_10);
	// 对应的n和m
	vector<int> index_n(G_ZMLIST_10.size(), 0);
	vector<int> index_m(G_ZMLIST_10.size(), 0);

	for (int i = 0; i < G_ZMLIST_10.size(); i++) {
		index_n[i] = G_ZMLIST_10[i][0];
		index_m[i] = G_ZMLIST_10[i][1];
	}
	// 找到合适的嵌入位置
	vector<int> index_suitable;

	for (int i = 0; i < G_ZMLIST_10.size(); i++) {
		// 10*10 embed range
		if (((index_m[i] == 0) && (index_n[i] < 0)) ||
			((index_m[i] > 0) && (index_m[i] <= 10) && (-4 <= index_n[i]) && (index_n[i] <= 4))) {
			index_suitable.push_back(i);
		}
		// 32*32 embed range
		//if (((index_m[i] == 0) && (index_n[i] < 0)) ||
		//	((index_m[i] > 0) && (index_m[i] <= 30) && (-16 <= index_n[i]) && (index_n[i] <= 16))) {
		//	index_suitable.push_back(i);
		//}
	}

	index_n.clear();
	index_m.clear();
	// 随机选出嵌入水印的位置
	vector<int> index_selected_embed(index_suitable.begin(), index_suitable.begin() + G_L);

	index_suitable.clear();
	// 得到在zmlist中的对应的数据，为下一步重构准备的
	G_ZMLIST_SELECTED_10.resize(G_L, vector<int>(2, 0));

	G_A_NM_SELECTED.resize(G_L, (0,0));

	for (int i = 0; i < G_L; i++) {
		G_ZMLIST_SELECTED_10[i] = G_ZMLIST_10[index_selected_embed[i]];
		G_A_NM_SELECTED[i] = G_A_NM_10[index_selected_embed[i]];
	}

	index_selected_embed.clear();

	G_A_NM_MODIFIED = G_A_NM_SELECTED;

	vector<double> III(G_L, 0.0);

	vector<double> IIII(G_L, 0.0);

	for (int i = 0; i < G_L; i++) {
		complex<double> A_selected = G_A_NM_MODIFIED[i];

		III[i] = ((-DD) + sqrt(pow(DD, 2) + 2 * EE * abs(A_selected))) / EE;

		double flag = 2 * floor((III[i] + 1) / 2);

		if (flag >= III[i]) {
			IIII[i] = 2 * floor((III[i] + 1) / 2) - w1[i];
		}
		else {
			IIII[i] = 2 * floor((III[i] + 1) / 2) + w1[i];
		}

		double aa = DD * IIII[i];

		double bb = EE * ((pow(IIII[i], 2)) * 0.5);

		G_A_NM_MODIFIED[i] = complex_sign(A_selected) * (aa + bb);
	}

	III.clear();
	IIII.clear();

	G_GP_REC_BEFOREMODIFY.resize(M, vector<complex<double>>(M, complex<double>(0, 0)));

	//cout << "\n go to zhishujurec622" << endl;

	zhishujurec622(G_A_NM_SELECTED, G_ZMLIST_SELECTED_10, M, oo, rr, G_GP_REC_BEFOREMODIFY);

	G_A_NM_SELECTED.clear();

	//cout << "--------------------------" << endl;

	G_GP_REC.resize(M, vector<complex<double>>(M, complex<double>(0, 0)));

	zhishujurec622(G_A_NM_MODIFIED, G_ZMLIST_SELECTED_10, M, oo, rr, G_GP_REC);

	G_A_NM_MODIFIED.clear();
	G_ZMLIST_SELECTED_10.clear();
	//水印图像
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < M; j++) {
			Gp_watermarked[i][j] = fp[i][j] - std::real(G_GP_REC_BEFOREMODIFY[i][j]) + std::real(G_GP_REC[i][j]);
		}
	}

	G_GP_REC_BEFOREMODIFY.clear();
	G_GP_REC_BEFOREMODIFY.clear();
	G_GP_REC.clear();

	cout << "rows: " << Gp_watermarked.size() << "; cols: " << Gp_watermarked[0].size() << endl;
}

void zhishujurec622(vector<complex<double>> A_nm, vector<vector<int>>& zmlist_selected, int M,
	vector<vector<double>> oo, vector<vector<double>> rr, vector<vector<complex<double>>> &result) {
	// Initialize matrices
	const int SIZE = 10;
	const int COUNT_SIZE = 57;
	vector<int> count(COUNT_SIZE, 0);
	vector<double> temp_oo_row = oo[0];
	//vector<double> temp_rr_col;
	vector<complex<double>> RBF1_col(M, complex<double>(0.0, 0.0));
	vector<complex<double>> RBF2_col(M, complex<double>(0.0, 0.0));
	// Iterate through the 32x32 grid
	for (int x = 0; x < SIZE; x++) {
		//start = clock();
		for (int y = 0; y < SIZE; y++) {
			int row = x * SIZE + y;
			int nj = zmlist_selected[row][0];
			int mc = zmlist_selected[row][1];
			complex<double> A_nm_value = A_nm[row];
			complex<double> conj_A_nm_value = conj(A_nm_value);
			int cnt_nj = count[nj + 20]; // 32*32 +40; 10*10 +20; 16*16 +20;
			double coefficient = 2 * nj * PI;
			for (int i = 0; i < M; i++)
			{
				double rr_val = rr[i][0]; // 行相同
				double image_rr_val = coefficient * rr_val;
				double cos_image_rr_val = cos(image_rr_val);
				double sin_image_rr_val = sin(image_rr_val);
				if (cnt_nj == 0)
				{
					double sqrt_part = sqrt(2.0 / rr_val);
					RBF1_col[i] = sqrt_part * complex<double>(cos_image_rr_val, sin_image_rr_val);
					RBF2_col[i] = sqrt_part * complex<double>(cos_image_rr_val, -sin_image_rr_val);
				}
				complex<double> coefficient1 = A_nm_value * RBF1_col[i];
				complex<double> coefficient2 = conj_A_nm_value * RBF2_col[i];
				for (int j = 0; j < M; j++)
				{
					double oo_val = temp_oo_row[j];// 列相同
					double image_rr_val = mc * oo_val;
					double cos_oo_val = cos(image_rr_val);
					double sin_oo_val = sin(image_rr_val);
					complex<double> front = coefficient1 * complex<double>(cos_oo_val, sin_oo_val);
					complex<double> behind = coefficient2 * complex<double>(cos_oo_val, -sin_oo_val);
					result[i][j] += front + behind;
				}
			}
			if (cnt_nj == 0)
			{
				count[nj + 20] = 1; // 32*32 +40; 10*10 +20; 16*16 +20;
			}
		}
	}
	
	// ========================================================================================================================
	//// 创建Eigen矩阵
	//MatrixXcd RBF1(M, M);
	//MatrixXcd RBF2(M, M);
	//MatrixXcd P_NM1(M, M);
	//MatrixXcd P_NM2(M, M);
	//// 填充RBF矩阵
	//RBF1.setZero();
	//RBF2.setZero();
	//for (int i = 0; i < M; i++) {
	//	for (int j = 0; j < M; j++) {
	//		double r = rr[i][j];
	//		double theta = oo[i][j];
	//		RBF(i, j) = sqrt(2.0 / r) * exp(complex<double>(0, theta));
	//	}
	//}
	//// 填充P_NM矩阵
	//P_NM1.setZero();
	//P_NM2.setZero();
	//for (size_t k = 0; k < zmlist_selected.size(); k++) {
	//	int n = zmlist_selected[k][0];
	//	int m = zmlist_selected[k][1];
	//	complex<double> a_nm = A_nm[k];
	//	for (int i = 0; i < M; i++) {
	//		for (int j = 0; j < M; j++) {
	//			double r = rr[i][j];
	//			P_NM(i, j) += a_nm * pow(r, abs(n)) * exp(complex<double>(0, m * oo[i][j]));
	//		}
	//	}
	//}
	//// 执行矩阵乘法
	//MatrixXcd Result = RBF.cwiseProduct(P_NM);
	//// 将结果转换回vector<vector<complex<double>>>格式
	//for (int i = 0; i < M; i++) {
	//	for (int j = 0; j < M; j++) {
	//		result[i][j] = Result(i, j);
	//	}
	//}
	//for (int x = 0; x < SIZE; x++) {
	//	for (int y = 0; y < SIZE; y++) {
	//		int row = x * SIZE + y;
	//		int nj = zmlist_selected[row][0];
	//		int mc = zmlist_selected[row][1];
	//		int cnt_nj = count[nj + 40];
	//		if (cnt_nj == 0)
	//		{
	//			for (int i = 0; i < M; i++) {
	//				for (int j = 0; j < M; j++) {
	//					double r = rr[i][j];
	//					double theta = 2 * nj * PI * r;
	//					RBF1(i, j) = sqrt(2.0 / r) * exp(complex<double>(0, theta));
	//					RBF2(i, j) = sqrt(2.0 / r) * exp(complex<double>(0, -theta));
	//				}
	//			}
	//		}
	//		for (int i = 0; i < M; i++) {
	//			for (int j = 0; j < M; j++) {
	//				double o = oo[i][j];
	//				double image_part = mc * o;
	//				P_NM1(i, j) = RBF1(i, j) * exp(complex<double>(0, image_part));
	//				P_NM2(i, j) = RBF2(i, j) * exp(complex<double>(0, -image_part));
	//				A_nm[row] * P_NM1(i, j) + P_NM2(i, j);
	//			}
	//		}
	//	}
	//}
	//cout << "zhishujurec622 finsh" << endl;
}

// 对于复数的sign函数，返回其实部和虚部的符号组成的复数  
template<typename T>

std::complex<T> complex_sign(const std::complex<T>& z) {
	// 处理输入为零的情况
	if (z == std::complex<T>(0, 0)) {
		return std::complex<T>(0, 0);
	}

	// 计算复数的模
	T magnitude = std::abs(z);

	// 返回归一化的复数
	return std::complex<T>(z.real() / magnitude, z.imag() / magnitude);
}

vector<double> wavelet(vector<double> BLOCK1, int level)
{
	vector<double> swt_output;

	int length = 0;

	swt(BLOCK1, level, "haar", swt_output, length);

	return swt_output;
}

void printSignal(vector<double> &sig, int length = 0)
{
	if (length == 0) length = sig.size();

	for (int i = 0; i < length; i++)
	{
		cout << sig[i] << " ";
	}

	cout << endl;
}

double calculateMean(vector<vector<double>>& yo, int start, int end, int column)
{
	double sum = 0.0;
	int count = 0;

	for (int i = start; i <= end; i++)
	{
		sum += yo[i][column];
		count++;
	}

	return sum / count;
}

vector<vector<int>> general(const vector<vector<int>> &w, int m, int n, int a, int b)
{
	cout << "in general" << endl;

	vector<vector<int>> f(m, vector<int>(n, 0));

	for (int x = 0; x < m; x++)
	{
		for (int y = 0; y < n; y++)
		{
			int x1 = (x + 1) + a * (y + 1);
			if (x1 >= m || x1 < 0)
			{
				x1 = (x1 % m + m) % m;
				if (x1 == 0) x1 = m;
			}
			f[x1 - 1][y] = w[x][y];
		}
	}

	vector<vector<int>> tmp = f;

	for (int x = 0; x < m; x++)
	{
		for (int y = 0; y < n; y++)
		{
			int y1 = b * (x + 1) + (y + 1);
			if (y1 >= n || y1 < 0)
			{
				y1 = (y1 % n + n) % n;
				if (y1 == 0) y1 = n;
			}
			f[x][y1 - 1] = tmp[x][y];
		}
	}

	return f;
}

vector<vector<int>> matToVector(const cv::Mat &ww, int rows, int cols)
{
	vector<vector<int>> ww_matrix(rows, vector<int>(cols, 0));

	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (ww.at<uchar>(i, j) > 0)
			{
				ww_matrix[i][j] = 1;
			}
			else {
				ww_matrix[i][j] = 0;
			}
		}
	}

	return ww_matrix;
}

void printIntSignal(vector<int>& signal, int length = 0)
{
	if (length == 0) length = signal.size();

	for (int i = 0; i < length; i++)
	{
		cout << signal[i] << "\t";
	}
	cout << endl;
}

void printMat(vector<vector<double>> mat)
{
	for (std::vector<double> sub_mat : mat) {
		for (double value : sub_mat) {
			cout << value << "\t";
		}
		cout << endl;
	}
	cout << endl;
}

void printIntMat(std::vector<std::vector<int>> mat)
{
	for (std::vector<int> sub_mat : mat) {
		for (int value : sub_mat) {
			cout << value << "  ";
		}
		cout << endl;
	}
	cout << endl;
}

vector<double> readWav(drwav& wav, int& wavelength, int& channels, int& Fs) {
	// 使用 dr_mp3 打开 MP3 文件
	//drwav wav;
	//if (!drwav_init_file(&wav, filename, nullptr)) {
	//	std::cerr << "Error opening WAV file." << std::endl;
	//	return {};
	//}

	// 获取音频信息
	channels = static_cast<int>(wav.channels);
	drwav_uint64 totalPCMFrameCount = wav.totalPCMFrameCount; // 获取总 PCM 帧数

	// 计算总样本数
	wavelength = static_cast<int>(totalPCMFrameCount);
	Fs = static_cast<int>(wav.sampleRate);

	// 创建存储样本的向量
	std::vector<float> audio_f(totalPCMFrameCount * channels);
	std::vector<double> audio(totalPCMFrameCount * channels);

	// 读取 PCM 数据
	drwav_read_pcm_frames_f32(&wav, totalPCMFrameCount, audio_f.data());

	// 关闭 WAV 文件
	drwav_uninit(&wav);

	// 将 float 转换为 double
	for (size_t i = 0; i < totalPCMFrameCount * channels; i++) {
		audio[i] = static_cast<double>(audio_f[i]);
	}

	return audio;
}

bool canFormSquareMatrix(const int wm_size) {
	if (wm_size < 0) return false; // 负数不能组成方阵
	int root = static_cast<int>(std::sqrt(wm_size));
	return (root * root == wm_size); // 检查是否为完全平方数
}

void getAudio20sFront(bool split, int max_wavelength, int Fs, int channels, int &wavelength, vector<double> yo_origin, vector<double> &yo)
{
	if (split)
	{
		yo.reserve(max_wavelength); // 预留空间以提高性能
		yo.assign(yo_origin.begin(), yo_origin.begin() + max_wavelength);
		wavelength = max_wavelength / channels;
	}
	else {
		yo = yo_origin;
	}
}

void splitAudioByTime(int max_wavelength, int Fs, int channels, int& wavelength, vector<double> yo_origin, vector<double>& yo, const double start_time, const double end_time, int start)
{
	yo.resize(max_wavelength);
	for (int i = 0; i < max_wavelength; ++i) {
		yo[i] = yo_origin[start + i];
	}
	wavelength = max_wavelength / channels;
}
