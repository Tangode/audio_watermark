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

    // ͨ����־��¼����
    template<typename T>
    void log(const T& message) {
        std::lock_guard<std::mutex> guard(logMutex);  // �̰߳�ȫ
        std::ostringstream oss;
        oss << message;
        logFile << oss.str() << std::endl;
    }

    // ���غ����Դ��� std::vector<int>
    void log(const std::vector<int>& vec) {
        std::lock_guard<std::mutex> guard(logMutex);  // �̰߳�ȫ
        for (size_t i = 0; i < vec.size(); ++i) {
            logFile << vec[i];
            if (i < vec.size() - 1) {
                logFile << " ";  // Ԫ��֮���ö��ŷָ�
            }
        }
        logFile << std::endl;  // ����һ��
    }

    // ���غ����Դ��� std::vector<double>
    void log(const std::vector<double>& vec) {
        std::lock_guard<std::mutex> guard(logMutex);  // �̰߳�ȫ
        for (size_t i = 0; i < vec.size(); ++i) {
            logFile << vec[i];
            if (i < vec.size() - 1) {
                logFile << " ";  // Ԫ��֮���ö��ŷָ�
            }
        }
        logFile << std::endl;  // ����һ��
    }

private:
    std::ofstream logFile;
    std::mutex logMutex;  // ������־�ļ��Ļ�����
};

#endif // LOGGER_H