#ifndef LOGGER_H
#define LOGGER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <mutex>

class Logger {
public:
    Logger(const std::string& filename = "audio_log.txt")
        : logFile(filename, std::ios::app) {
        if (!logFile.is_open()) {
            throw std::runtime_error("Unable to open log file.");
        }
    }

    ~Logger() {
        if (logFile.is_open()) {
            logFile.close();
        }
    }

    // 通用日志记录函数
    template<typename T>
    void log(const T& message) {
        std::lock_guard<std::mutex> guard(logMutex);  // 线程安全
        std::ostringstream oss;
        oss << message;
        logFile << oss.str() << std::endl;
    }

    // 重载函数以处理 std::vector<int>
    void log(const std::vector<int>& vec) {
        std::lock_guard<std::mutex> guard(logMutex);  // 线程安全
        for (size_t i = 0; i < vec.size(); ++i) {
            logFile << vec[i];
            if (i < vec.size() - 1) {
                logFile << " ";  // 元素之间用逗号分隔
            }
        }
        logFile << std::endl;  // 结束一行
    }

    // 重载函数以处理 std::vector<double>
    void log(const std::vector<double>& vec) {
        std::lock_guard<std::mutex> guard(logMutex);  // 线程安全
        for (size_t i = 0; i < vec.size(); ++i) {
            logFile << vec[i];
            if (i < vec.size() - 1) {
                logFile << " ";  // 元素之间用逗号分隔
            }
        }
        logFile << std::endl;  // 结束一行
    }

private:
    std::ofstream logFile;
    std::mutex logMutex;  // 保护日志文件的互斥量
};

#endif // LOGGER_H