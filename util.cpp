#include <vector>
#include <fftw3.h>
#include <complex>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <mpg123.h>
#include <filesystem>
#include <string>
#include <windows.h>
#include <algorithm>
#include <lame/lame.h>
#include <cstdint>
#include <soxr.h>
#include "global.h"
#include "dr_wav.h"
#include "util.h"
#include "dr_mp3.h"
#include "miniaudio.h"
#include "embed.h"

using namespace std;
using namespace cv;


const double PI = 3.1416;

vector<double> tto(vector<vector<double>> &I, int N) {
    // 获取二维数组的行数和列数
    int rows = I.size();
    int cols = I[0].size();

    // 创建一维数组 K 来存储结果
    vector<double> K;

    // 将二维数组 I 转换为一维数组 K
    for (int p = 0; p < N; ++p) {
        for (int q = 0; q < N; ++q) {
            K.push_back(I[p][q]);
        }
    }

    return K;
}

vector<int> int_tto(vector<vector<int>> &I, int N)
{
	int rows = I.size();
	int cols = I[0].size();

	vector<int> K;

	for (int p = 0; p < N; p++)
	{
		for (int q = 0; q < N; q++)
		{
			K.push_back(I[p][q]);
		}
	}

	return K;
}

vector<vector<double>> ott(const vector<double>& I, int N)
{
	vector<vector<double>> K;

	for (int p = 0; p < N; p++)
	{
		vector<double> temp_row;

		for (int q = 0; q < N; q++)
		{
			temp_row.push_back(I[p * N + q]);
		}

		K.push_back(temp_row);
		temp_row.clear();
	}

	return K;
}

vector<vector<int>> int_ott(const vector<int>& I, int N) {
    // 初始化结果矩阵 K
    vector<vector<int>> K(N, vector<int>(N, 0));
    // 进行矩阵赋值
    for (int p = 0; p < N; ++p) {
        for (int q = 0; q < N; ++q) {
            K[p][q] = I[p * N + q];
        }
    }
    // 返回结果矩阵 K
    return K;
}

// 1D FFT function
void fft1d(vector<complex<double>>& data, bool invert) {
    int n = data.size();
    if (n <= 1) return;

    vector<complex<double>> even(n / 2);
    vector<complex<double>> odd(n / 2);

    for (int i = 0; i < n / 2; i++) {
        even[i] = data[i * 2];
        odd[i] = data[i * 2 + 1];
    }

    fft1d(even, invert);
    fft1d(odd, invert);

    double angle = 2 * PI / n * (invert ? -1 : 1);
    complex<double> w(1), wn(cos(angle), sin(angle));

    for (int i = 0; i < n / 2; i++) {
        complex<double> t = w * odd[i];
        complex<double> u = even[i];
        data[i] = u + t;
        data[i + n / 2] = u - t;
        w *= wn;
    }

    if (invert) {
        for (auto& x : data) x /= n;
    }
}

void fft2(Mat& input, Mat& output, bool invert = false, bool normalize = false)
{
    int rows = input.rows;
    int cols = input.cols;

    // Perform 1D FFT on each row
    for (int i = 0; i < rows; i++) {
        vector<complex<double>> row(cols);
        for (int j = 0; j < cols; j++) {
            row[j] = complex<double>(input.at<double>(i, j), 0);
        }
        fft1d(row, invert);
        for (int j = 0; j < cols; j++) {
            output.at<Vec2d>(i, j) = Vec2d(row[j].real(), row[j].imag());
        }
    }

    // Perform 1D FFT on each column
    for (int j = 0; j < cols; j++) {
        vector<complex<double>> col(rows);
        for (int i = 0; i < rows; i++) {
            col[i] = complex<double>(output.at<Vec2d>(i, j)[0], output.at<Vec2d>(i, j)[1]);
        }
        fft1d(col, invert);
        for (int i = 0; i < rows; i++) {
            output.at<Vec2d>(i, j) = Vec2d(col[i].real(), col[i].imag());
        }
    }

    // Normalize the result if required
    //if (normalize && invert) {
    //    double scale = 1.0 / (rows * cols);
    //    for (int i = 0; i < rows; ++i) {
    //        for (int j = 0; j < cols; ++j) {
    //            Vec2d& value = output.at<Vec2d>(i, j);
    //            value[0] *= scale;
    //            value[1] *= scale;
    //        }
    //    }
    //}
}

void zhishujufenjie62(vector<vector<double>>& Gp, int M, int Nmax, vector<complex<double>>& A_nm, vector<vector<int>>& zmlist) {
    int rows = Gp.size();
    int cols = Gp[0].size();
    const int size = 2 * Nmax + 1;
    // Allocate FFTW complex arrays
    fftw_complex* in = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    fftw_complex* out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * rows * cols);
    // Copy input data to FFTW format
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            in[i * cols + j][0] = Gp[i][j]; // Real part
            in[i * cols + j][1] = 0.0;      // Imaginary part
        }
    }
    // Create FFTW plan
    //fftw_plan p = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_plan plan1 = fftw_plan_dft_2d(rows, cols, in, out, FFTW_FORWARD, FFTW_ESTIMATE);
    //fftw_plan col_plan = fftw_plan_dft_2d(cols, rows, out, in, FFTW_FORWARD, FFTW_ESTIMATE);
    // Execute FFT
    fftw_execute(plan1);
    //fftw_execute(col_plan);
    // Copy FFT result to output vector
    vector<vector<complex<double>>> Gp_fft(rows, vector<complex<double>>(cols, complex<double>(0, 0)));
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            Gp_fft[i][j] = complex<double>(out[i * cols + j][0], in[i * cols + j][1]);
        }
    }
    // Cleanup
    fftw_destroy_plan(plan1);
    //fftw_destroy_plan(col_plan);
    fftw_free(in);
    fftw_free(out);
    vector<vector<complex<double>>> Ekm(M, vector<complex<double>>(M, complex<double>(0, 0)));
    double normalization = M * M;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < M; j++) {
            Ekm[i][j] = Gp_fft[i][j] / normalization;
        }
    }
    // 初始化 E
    vector<vector<complex<double>>> E(size, vector<complex<double>>(size, complex<double>(0, 0)));
    // 填充 E
    for (int i = 0; i < Nmax; i++) {
        for (int j = 0; j < Nmax; j++) {
            E[i][j] = Ekm[M - Nmax + i][M - Nmax + j];
        }
        for (int j = 0; j <= Nmax; j++) {
            E[i][Nmax + j] = Ekm[M - Nmax + i][j];
        }
    }
    for (int i = 0; i <= Nmax; i++) {
        for (int j = 0; j < Nmax; j++) {
            E[Nmax + i][j] = Ekm[i][M - Nmax + j];
        }
        for (int j = 0; j <= Nmax; j++) {
            E[Nmax + i][Nmax + j] = Ekm[i][j];
        }
    }
    int c = 0;
    for (int k = 0; k < size; k++) {
        for (int m = 0; m < size; m++) {
            zmlist[c][0] = k - Nmax;
            zmlist[c][1] = m - Nmax;
            A_nm[c] = E[k][m];
            c++;
        }
    }
}

vector<int16_t> floatToPCM16(vector<double>& input, float scaleFactor = 32767.0f) {
    vector<int16_t> output(input.size());
    for (int i = 0; i < input.size(); i++) {
        // 将浮点数转换为16位整数，注意避免溢出  
        int16_t value = static_cast<int16_t>(input[i] * scaleFactor);
        if (value > 32767) value = 32767;
        if (value < -32768) value = -32768;
        output[i] = value;
    }
    return output;
}

double getGlobalProgress()
{
    return g_audio_wm_progress;
}

void save_audio_drwav(const wchar_t* outputFilename, vector<double>& yo, int sampleRate, int channels)
{
    std::vector<int16_t> audioData(yo.size());
    for (size_t i = 0; i < yo.size(); ++i) {
        audioData[i] = static_cast<int16_t>(max(-1.0, min(1.0, yo[i])) * 32767);
    }

    drwav wav;
    const drwav_data_format drwav_data_format{
        drwav_container_riff,
        DR_WAVE_FORMAT_PCM,
        static_cast<uint16_t>(channels),
        static_cast<uint32_t>(sampleRate),
        16
    };

    // 调试输出路径
    std::wcout << L"Output path: " << outputFilename << std::endl;

    if (!drwav_init_file_write_w(&wav, outputFilename, &drwav_data_format, nullptr)) {
        std::cerr << "Failed to open WAV file for writing: " << outputFilename << std::endl;
        return;
    }

    drwav_write_pcm_frames(&wav, static_cast<drwav_uint64>(yo.size() / channels), audioData.data());
    drwav_uninit(&wav);
}

void save_audio_drmp3(const wchar_t* outputFilename, vector<double>& yo, int sampleRate, int channels)
{
    // outputFilename 是文件保存路径， yo 是double音频信号， sampleRate 是采样率， channels 是声道数， 用lame库保存mp3 文件
    // 创建LAME编码器
    lame_t lame = lame_init();
    lame_set_in_samplerate(lame, sampleRate);
    lame_set_num_channels(lame, channels);
    lame_set_out_samplerate(lame, sampleRate);
    lame_set_quality(lame, 5); // 0-9, 5是中等质量
    lame_init_params(lame);

    // 打开输出文件
    FILE* mp3File = _wfopen(outputFilename, L"wb");
    if (!mp3File) {
        std::cerr << "cann't open file: " << outputFilename << std::endl;
        lame_close(lame);
        return;
    }

    // Convert yo from double to short
    std::vector<short> pcmBuffer(yo.size());
    for (size_t i = 0; i < yo.size(); ++i) {
        pcmBuffer[i] = static_cast<short>(yo[i] * 32767.0); // Convert to 16-bit PCM
    }

    // Buffer for MP3 data
    const int MP3_BUFFER_SIZE = 1.25 * pcmBuffer.size() + 7200; // LAME recommendation
    std::vector<unsigned char> mp3Buffer(MP3_BUFFER_SIZE);

    // Encode PCM to MP3
    int mp3Bytes;
    if (channels == 1) {
        mp3Bytes = lame_encode_buffer(lame, pcmBuffer.data(), nullptr, pcmBuffer.size(), mp3Buffer.data(), MP3_BUFFER_SIZE);
    }
    else {
        mp3Bytes = lame_encode_buffer_interleaved(lame, pcmBuffer.data(), pcmBuffer.size() / channels, mp3Buffer.data(), MP3_BUFFER_SIZE);
    }

    if (mp3Bytes < 0) {
        std::cerr << "LAME encoding error: " << mp3Bytes << std::endl;
        fclose(mp3File);
        lame_close(lame);
        return;
    }

    // Write MP3 data to file
    fwrite(mp3Buffer.data(), 1, mp3Bytes, mp3File);

    // Flush LAME buffer
    mp3Bytes = lame_encode_flush(lame, mp3Buffer.data(), MP3_BUFFER_SIZE);
    fwrite(mp3Buffer.data(), 1, mp3Bytes, mp3File);

    // Clean up
    fclose(mp3File);
    lame_close(lame);
}

string Widen(const wchar_t* wstr) {
    if (!wstr) return {}; // 检查空指针

    // 获取所需的字符数
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);

    // 创建足够大的字符串
    std::string str(size_needed, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &str[0], size_needed, nullptr, nullptr);

    return str;
}

vector<double> mp3ToDoubleVector(const char* mp3File, int &channels, long &rate, int &wavelength)
{
    mpg123_init();
    mpg123_handle* mh = mpg123_new(NULL, NULL);
    if (mh == nullptr) {
        throw std::runtime_error("Failed to create mpg123 handle");
    }

    if (mpg123_open(mh, mp3File) != MPG123_OK) {
        mpg123_delete(mh);
        throw std::runtime_error("Failed to open MP3 file");
    }

    int encoding;

    mpg123_getformat(mh, &rate, &channels, &encoding);

    std::vector<double> audioData;
    unsigned char buffer[4096];
    size_t done;

    while (mpg123_read(mh, buffer, sizeof(buffer), &done) == MPG123_OK) {
        size_t numSamples = done / sizeof(short);
        audioData.resize(audioData.size() + numSamples);

        // Convert to double and store in vector
        for (size_t i = 0; i < numSamples; ++i) {
            audioData[audioData.size() - numSamples + i] =
                static_cast<double>(reinterpret_cast<short*>(buffer)[i]) / SHRT_MAX;
        }
    }

    mpg123_close(mh);
    mpg123_delete(mh);
    mpg123_exit();

    wavelength = audioData.size() / channels;

    return audioData;
}

bool isMp3File(fs::path& path, string fpath) 
{
    // 检查文件是否存在
    if (!fs::exists(fpath)) {
        std::cerr << "File does not exist: " << std::endl;
        return false;
    }
    // 检查文件扩展名
    return path.extension() == ".mp3" || path.extension() == ".MP3";
}

string toUTF8(const std::wstring& wstr) {
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), nullptr, 0, nullptr, nullptr);
    std::string str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.c_str(), static_cast<int>(wstr.size()), &str[0], size_needed, nullptr, nullptr);
    return str;
}

wstring toWideString(const string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), nullptr, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size_needed);
    return wstr;
}

void modifyAndCombinePath(const char* path, const char* save_path, char* output) {
    // 提取文件名
    const char* filename = strrchr(path, '/'); // 找到最后一个 '/'
    if (!filename) {
        filename = strrchr(path, '\\'); // 如果是 Windows 路径
    }

    if (filename) {
        filename++; // 跳过 '/'
    }
    else {
        filename = path; // 如果没有找到 '/', 使用整个路径
    }

    // 拼接新的文件名
    char newFileName[256]; // 假设文件名长度不会超过 255
    snprintf(newFileName, sizeof(newFileName), "%.*s_wm_release.wav",
        (int)(strrchr(filename, '.') - filename), filename);

    // 拼接保存路径和新文件名
    snprintf(output, 512, "%s%s", save_path, newFileName);
}

string ConvertUTF16ToMBCS(const wchar_t* utf16Str)
{
    // 获取转换后的字符串长度
    int sLen = WideCharToMultiByte(CP_ACP, 0, utf16Str, -1, NULL, 0, NULL, NULL);
    if (sLen == 0)
    {
        std::cerr << "Error converting UTF-16 to system encoding" << std::endl;
        return "";
    }

    std::string str(sLen, 0);
    WideCharToMultiByte(CP_ACP, 0, utf16Str, -1, &str[0], sLen, NULL, NULL);
    return str;
}

// 读取 WAV 文件
bool transResampleReadWAV(const char* inputPath, vector<float>& buffer, int& sampleRate, int& channels) {
    SF_INFO sfInfo;
    SNDFILE* infile = sf_open(inputPath, SFM_READ, &sfInfo);
    if (!infile) {
        std::cerr << "Error opening input file: " << sf_strerror(infile) << std::endl;
        return false;
    }
    channels = sfInfo.channels;
    sampleRate = sfInfo.samplerate;
    buffer.resize(sfInfo.frames * sfInfo.channels);
    sf_count_t readFrames = sf_readf_float(infile, buffer.data(), sfInfo.frames);
    sf_close(infile);

    // Resize buffer to the actual number of read frames
    buffer.resize(readFrames);

    return true;
}

// 写入 WAV 文件
bool transResampleWriteWAV(const char* outputPath, const vector<float>& buffer, double sampleRate, int channels) {
    SF_INFO sfInfo{};
    sfInfo.frames = buffer.size() / channels; // Assuming stereo (2 channels)
    sfInfo.samplerate = static_cast<int>(sampleRate);
    sfInfo.channels = channels; // Assuming stereo
    sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    SNDFILE* outfile = sf_open(outputPath, SFM_WRITE, &sfInfo);
    if (!outfile) {
        cerr << "Error opening output file: " << sf_strerror(outfile) << endl;
        return false;
    }

    sf_count_t writtenFrames = sf_writef_float(outfile, buffer.data(), sfInfo.frames);
    sf_close(outfile);

    if (writtenFrames != sfInfo.frames) {
        cerr << "Error writing WAV file: " << outputPath << endl;
        return false;
    }

    return true;
}

// 重采样音频数据
void resample(const float* inputBuffer, size_t inputSize,
    float* outputBuffer, size_t& outputSize,
    double inputRate, double outputRate, int channels) {
    soxr_error_t error;

    // 创建重采样上下文
    soxr_t soxr = soxr_create(inputRate, outputRate, channels, &error, nullptr, nullptr, nullptr);
    if (error) {
        std::cerr << "Error creating soxr: " << error << std::endl;
        return;
    }

    // 进行重采样
    size_t outputFrames = outputSize;
    error = soxr_process(soxr, inputBuffer, inputSize, nullptr,
        outputBuffer, outputFrames, &outputSize);

    if (error) {
        cerr << "Error during resampling: " << error << endl;
    }

    // 清理
    soxr_delete(soxr);
}

void convert_wav(vector<double>& inputBuffer, int& inputRate, int channels, const wchar_t* outputPath) {
    cout << "Input buffer size: " << inputBuffer.size() << std::endl;
    double outputRate = 44100; // 目标采样率
    size_t inputSize = inputBuffer.size();
    size_t outputSize = static_cast<size_t>(static_cast<double>(inputSize) * outputRate / inputRate) + 1;
    vector<float> outputBuffer(outputSize);

    vector<float> in;
    for (auto i : inputBuffer) {
        in.push_back((float)i);
    }
    //// 调用重采样函数
    resample(in.data(), inputSize, outputBuffer.data(), outputSize, inputRate, outputRate, channels);

    // 写入输出 WAV 文件
    vector<double> out;
    for (auto i : outputBuffer) {
        out.push_back((double)i);
    }
    inputBuffer = out;
    inputRate = outputRate;
    //save_audio_drwav(outputPath, out, outputRate, channels);
    return;
}

wstring AddSuffixToPath(const wchar_t* originalPath, const wstring& suffix) {
    wstring path(originalPath);

    // 查找最后一个点的位置
    size_t dotPosition = path.find_last_of(L".");

    // 如果找到了点，插入 suffix
    if (dotPosition != wstring::npos) {
        path.insert(dotPosition, suffix);
    }
    else {
        // 如果没有找到点，直接在末尾添加 suffix
        path.append(suffix);
    }

    return path;
}

wstring string_to_wstring(const std::string& str) {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str.c_str(), (int)str.size(), &wstr[0], size_needed);
    return wstr;
}