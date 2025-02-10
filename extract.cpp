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

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

int wm_extract(const wchar_t *path, int *wm)
{
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
		//drwav wav;
		drwav wav;
		if (!drwav_init_file(&wav, extract_path.c_str(), nullptr)) {
			std::cerr << "Error opening WAV file." << std::endl;
			return 1;
		}

		yw = readWav(wav, wavelength, numChannels, Fs);
	}

	//int max_wavelength = Fs * 25 * numChannels;
	//bool split = yw_origin.size() > max_wavelength;
	//getAudio20sFront(split, max_wavelength, Fs, numChannels, wavelength, yw_origin, yw);

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

	int k1 = 5;
	double PI = 3.1416;
	double D = 0.8;
	double DD = 0.0012;
	double EE = 0.00002;
	//int blocklength = pp * qq * 8 * 8; // 32x32 ˮӡ��С
	int blocklength = 65536;
	int k = 0;
	int i = 1;
	int rela = 1;
	vector<int> t0;

	cout << "wavelength: " << wavelength << endl;

	while (k + (i * Lsyn * k1) < wavelength) {
		while (rela != 0 && (k + Lsyn * k1) < wavelength) {
			for (int mm = 0; mm < Lsyn; mm++) {
				double tempmean = 0.0;

				int front = k + mm * k1 + 1;

				int back = k + (mm + 1) * k1;

				tempmean = calculateMean(audio_data, front, back, 0);

				int temp = floor(tempmean / D);

				synw[mm] = (temp % 2 + 2) % 2;

				temp1[mm] = syn[mm] ^ synw[mm];
			}

			rela = accumulate(temp1.begin(), temp1.end(), 0);

			k++;
		}
		t0.push_back(k);
		rela = 1;

		k = k + Lsyn * k1 + blocklength - 1;
		i++;
	}
	
	// ��ȡˮӡ��Ϣ
	vector<double> bt_temp;
	vector<vector<double>> bt;
	vector<int> t1;
	cout << "t0:" << endl;
	printIntSignal(t0, 0);
	for (int ii = 0; ii < i - 3; ii++) {
		t1.push_back(t0[ii] + Lsyn * k1);
		if (t1[ii] + blocklength - 1 > wavelength) {
			W1 = vector<vector<int>>(pp, vector<int>(qq, 1));
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

			if (n % 2 != 0) {
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

					Gp_w[u][v] = (I_w1[row][col]) * sqrt(static_cast<double>(u) / (2 * M));
				}
			}
			
			bt_temp = PQIMfft_extract(Gp_w, b, M, DD, EE);

			bt.push_back(bt_temp);
		}
	}
	cout << "t1:" << endl;
	printIntSignal(t1, t1.size());
	// watermark capacity 32*32=1024 16*16=256 10*10=100
	vector<int> r(100);
	for (int iii = 0; iii < 100; iii++) {
		double sum_bt = 0.0;

		// ���� bt ���к�
		for (int j = 0; j < bt.size(); j++) {
			sum_bt += bt[j][iii];
		}

		// ���� r(iii)
		r[iii] = static_cast<int>(round(sum_bt / (bt.size() - 1)));
	}

	for (size_t i = 0; i < r.size(); ++i) {
		wm[i] = r[i];
	}

	W1 = int_ott(r, 10);

	vector<vector<int>> W2 = igeneral(W1, pp, qq, 5, 6);

	cout << "\n W1:" << endl;
	printIntMat(W1);
	cout << "\n W2:" << endl;
	printIntMat(W2);
	//cout << "\n W:" << endl;
	//printIntMat(W);

	vector<int> wm_vec = int_tto(W2, 10);

	//wm = new int[wm_vec.size()];

	std::copy(wm_vec.begin(), wm_vec.end(), wm);

	//double BER = drzh(W, W2, pp, qq);

	//cout << "BER:" << BER << endl;
	//wmLog("wm_extract end");
	return 0;

	// todo: ͼ�񱣴�

	//string fname_origin = "D://wm_project/audio/female.mp3";
	////��ȡ��Ƶ�ļ�
	//SNDFILE* sndFile_origin;
	//SF_INFO sfInfo_origin{};
	//sfInfo_origin.format = 0;

	//sndFile_origin = sf_open(fname_origin.c_str(), SFM_READ, &sfInfo_origin);

	//if (!sndFile_origin) {
	//	cerr << "Error opening audio file." << endl;
	//}

	//int origin_numChannels = sfInfo_origin.channels;

	//vector<double> yo;

	//yo.resize(sfInfo_origin.frames * origin_numChannels);

	//sf_read_double(sndFile_origin, yo.data(), sfInfo_origin.frames * numChannels);

	//sf_close(sndFile_origin);

	//double psnr = psnrzh(yw, yo);

	//cout << "psnr:" << psnr << endl;

	//double wsr = getWsr(yw, yo);

	//cout << "wsr:" << wsr << endl;
}

vector<double> PQIMfft_extract(vector<vector<double>>& Gp_w, vector<int>& b, int M, double DD, double EE)
{
	//const vector<double> key1 = { 0.928193675245596, 0.762097922072739, 0.380267928722993, 0.897730320468743, 0.100822133636870, 0.0899504433067866, 0.239020109274386,
	//0.946779913782277, 0.0952670768819904, 0.320156919270540, 0.577931144724556, 0.196373121839455, 0.461274061204655, 0.323929634780567, 0.807739985547207, 0.540754730983380,
	//0.985142082650825, 0.958345377953601, 0.185229603238548, 0.0660488670603961, 0.00835748882788212, 0.433219741512797, 0.501328268578626, 0.684454533009209, 0.0274337980435039,
	//0.821913355398633, 0.758935114048920, 0.612561055737784, 0.223044160578087, 0.578077449230608, 0.287499282769805, 0.965226733423526, 1.11022302462516e-16, 0, 2.55749847299924e-07 };

	//const vector<double> key2 = { 0.164635665117893, 0.758726063009854, 0.00591829393010890, 0.964050036774166, 0.984922993786984, 0.976826357531615, 0.356700569394344,
	//0.203547152614262, 0.336560412745046, 0.498445989825352, 0.997342009306155, 0.749864280396276, 0.470142180541784, 0.874301822643799, 0.0868517482511835, 0.895214310591549,
	//0.583643833047015, 0.694493484576109, 0.884424240366055, 0.919066545678925, 0.790112074974144, 0.647724940400028, 0.363864662192446, 0.879399867540293, 0.249724426967429,
	//0.134744306769869, 0.248274963641751, 0.0776257270260381, 0.143523293499973, 0.263989850166916, 0.892556319897704, 0.333508113443291, 1.11022302462516e-16, 3.33066907387547e-15,
	//4.52511162984948e-07 };

	//int L = b.size();
	int L = 100;

	const int Nmax = 20;
	const int size = 2 * Nmax + 1;
	const int total_elements_32 = size * size;

	vector<complex<double>> A_nm(total_elements_32, (0, 0));
	vector<vector<int>> zmlist(total_elements_32, vector<int>(2, 0));
	// ָ���صķֽ�
	zhishujufenjie62(Gp_w, M, Nmax, A_nm, zmlist);

	vector<int> index_selected_extract(L);
	// �����������
	vector<int> index_suitable;

	vector<complex<double>> A_nm_selected_extract(L, 0.0);

	vector<double> b_extract(L, 0.0);

	for (int i = 0; i < zmlist.size(); i++) {
		int index_n_temp = zmlist[i][0];
		int index_m_temp = zmlist[i][1];

		// 10*10/16*16 embed range
		if (((index_m_temp == 0) && (index_n_temp < 0)) ||
			((index_m_temp > 0) && (index_m_temp <= 10) && (-4 <= index_n_temp) && (index_n_temp <= 4))) {
			index_suitable.push_back(i);
		}
		// 32*32 embed range
		/*if (((index_m_temp == 0) && (index_n_temp < 0)) ||
			((index_m_temp > 0) && (index_m_temp <= 30) && (-16 <= index_n_temp) && (index_n_temp <= 16))) {
			index_suitable.push_back(i);
		}*/
	}

	for (int i = 0; i < L; i++)
	{
		A_nm_selected_extract[i] = A_nm[index_suitable[i]];
	}

	for (int i = 0; i < L; i++)
	{
		long double magnitude = 2.0 * EE * abs(A_nm_selected_extract[i]);
		long double sqrt_part = sqrt(DD * DD + magnitude);
		long double A = (-DD) + sqrt_part;
		long double temp = A / EE;

		b_extract[i] = fmod(floor(temp + 0.5), 2.0);
	}

	return b_extract;
}

vector<vector<int>> igeneral(vector<vector<int>> W1, int m, int n, int a, int b)
{
	vector<vector<int>> f(m, vector<int>(n, 0));
	// ��һ����ѭ��
	for (int x = 0; x < m; x++) {
		for (int y = 0; y < n; y++) {
			int y1 = b * (x + 1) + (y + 1);

			if (y1 > n || y1 < 0) {
				y1 = (y1 % n + n) % n;
			}

			if (y1 == 0)
			{
				y1 = n;
			}

			f[x][y] = W1[x][y1 - 1];
		}
	}

	vector<vector<int>> tmp = f;
	// �ڶ�����ѭ��
	for (int x = 0; x < m; x++) {
		for (int y = 0; y < n; y++) {
			int x1 = (x + 1) + a * (y + 1);

			if (x1 > m || x1 < 0) {
				x1 = (x1 % m + m) % m;
			}

			if (x1 == 0)
			{
				x1 = m;
			}

			f[x][y] = tmp[x1 - 1][y];
		}
	}

	return f;
}

double drzh(const vector<vector<int>>& W, vector<vector<int>>& W1, int M, int N)
{
	double x = 0.0;
	vector<vector<double>> Wa(M, vector<double>(N, 0.0));
	vector<vector<double>> Wb(M, vector<double>(N, 0.0));
	// ��һ����ѭ��
	for (int i = 0; i < M; ++i) {
		for (int j = 0; j < N; ++j) {
			if (static_cast<double>(W[i][j]) != 0.0) {
				Wa[i][j] = 1.0;
			}
			if (W1[i][j] != 0.0) {
				Wb[i][j] = 1.0;
			}
			x += std::abs(Wa[i][j] - Wb[i][j]);
		}
	}
	// �������ս��
	double A = x / (M * N);

	return A;
}

double psnrzh(const vector<double>& M, const vector<double>& N)
{
	int a = std::min(M.size(), N.size()); // ���� a���� min(length(M), length(N))
	double sum = 0.0;

	// ��һ����ѭ��
	for (int i = 0; i < a; ++i) {
		double t = M[i]; // ԭ��Ƶ�ź� M
		double s = N[i];    // ��ˮӡ����Ƶ�ź� N
		double p = t - s;
		double q = p * p;
		sum += q;
	}

	double mse = sum / a; // ���������� MSE
	double psnr = 10 * std::log10(1.0 / mse); // ���� PSNR ֵ

	return psnr;
}

double getWsr(const vector<double>& M, const vector<double>& N)
{
	int a = std::min(M.size(), N.size()); // ���� a���� min(length(M), length(N))
	double sum = 0.0;
	double sum1 = 0.0;

	// ��һ����ѭ��
	for (int i = 0; i < a; ++i) {
		double t = M[i]; // ԭ��Ƶ�ź� M
		double s = N[i];    // ��ˮӡ����Ƶ�ź� N
		double p = t - s;
		double q = p * p;
		sum += q;
		sum1 += t * t;
	}

	// ���������� MSE ������� SNR
	double mse = sum / a;
	double mse_original = sum1 / a;
	double snr = 10 * std::log10(mse_original / mse);

	return snr;
}