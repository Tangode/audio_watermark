#include "util.h"
#include "embed.h"
#include "extract.h"
#include <fmt/core.h>
#include <opencv2/opencv.hpp>
#include <complex>
#include <thread>
#include "global.h"
#include "Log.h"
#include "new_embed.h"
#include "new_extract.h"
#include <iostream>
#include <string>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

int main(){
    //Log audio_logger;
    fmt::println("welcome to use audio watermark system");
    //audio_logger.log();
    //const wchar_t* path = L"D://音频/female.mp3";
    const wchar_t* path = L"D:/resource/error_file/output_002.mp3";
    const wchar_t* save_path = L"D:/resource/output_002_wm.mp3";
    const int wm[1024]{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0,
            0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1,
            1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1,
            1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0,
            0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    //const int wm[100] = {0,1,0,0,0,0,1,1,0,1,0,0,1,1,0,1,0,1,0,1,1,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,1,1,0,1,1,1,0,0,0,1,1,0,1,1,1,1,1,0,1,1,0,1,0,0,0,1,1,0,1,1,0,1,1,0,1,0,0,0,1,1,0,1,1,0,1};
    //wstring s_path = AddSuffixToPath(path, L"_resample");
    //wcout << s_path << endl;
    //convert_wav(path, s_path.c_str());
    //fs::path fname = s_path;
    //std::filesystem::remove(fname);
    int flag = 1;
    if (flag == 1)
    {
        //// 创建并启动线程
        //std::thread embedThread(embedWatermark);
        //// 主线程循环打印进度
        //while (embedThread.joinable()) {
        //    std::cout << "current audio embed progress: " << g_audio_wm_progress << "%" << std::endl;
        //    std::this_thread::sleep_for(std::chrono::seconds(30));  // 每500毫秒打印一次
        //}
        // 等待线程完成
        //embedThread.join();
         clock_t start = 0, end = 0;
         double totaltime = 0.0;
         start = clock();
         wm_embed(path, save_path, wm, 1024, 0, 25);
         //char* wm = "0000000000000000000000000000000000000000000000110000000000000000000000000000011100000000000000000000";
         //char* wm = "0000000100000000000000110000000000000010001100000001001111110000000110111111100000001111111110000000";
         //new_wm_embed(path, save_path, wm, 100, 0, 25);
         end = clock();
         totaltime = (double)(end - start) / CLOCKS_PER_SEC;
         cout << "耗时: " << totaltime << endl;
    }
    else {
        //const wchar_t* extract_path = L"D:/temp/apple_wm.wav";
        int* ex_wm = new int[1024];
        //char* wm = new char[100];
        //D://wm_audio/news_wm_release.wav
        int code = wm_extract(save_path, ex_wm, 1024);
        //int code = new_wm_extract(save_path, wm);
        int ber = 0;
        vector<int> tm;
        for (int i = 0; i < 1024; i++) {
            if (wm[i] != ex_wm[i]) {
                tm.push_back(i);
                ber++;
            }
        }
        cout << "be = " << ber << "; ber = " << static_cast<double>(ber) / 1024 * 100 << endl;
        for (auto item : tm) {
            cout << item << "\t";
        }
    }
    return 0;
}



//int main() {
//    string filePath = "D:/FFOutput/Souleymane.wav";
//    // 声明输出文件名格式
//    std::string outputFormat = "D:/FFOutput/Souleymane_%03d.wav"; // 输出文件名格式
//
//    // 使用 ffmpeg 拆分音频的命令
//    std::string command = "ffmpeg -i \"" + filePath + "\" -f segment -segment_time 30 -c copy \"" + outputFormat + "\"";
//
//    // 执行命令
//    int result = std::system(command.c_str());
//
//    // 检查执行结果
//    if (result != 0) {
//        std::cerr << "Error: Unable to split audio file." << std::endl;
//    }
//    else {
//        std::cout << "Audio file split into segments successfully." << std::endl;
//    }
//}