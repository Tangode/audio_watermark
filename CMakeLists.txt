# CMakeList.txt: wm_project 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("wm_project")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR D://wm_project/vcpkg_installed/x64-windows/share/opencv4)

set(CMAKE_TOOLCHAIN_FILE "D:/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

include_directories(D:/wm_project/out/build/x64-release/vcpkg_installed/x64-windows/include)

link_directories(D:/wm_project/out/build/x64-release/vcpkg_installed/x64-windows/lib)

find_package(fmt CONFIG REQUIRED)

find_package(OpenCV REQUIRED)

find_package(FFTW3 CONFIG REQUIRED)

find_package(FFTW3f CONFIG REQUIRED)

find_package(FFTW3l CONFIG REQUIRED)

find_package(SndFile CONFIG REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)

find_package(mpg123 CONFIG REQUIRED)

find_package(mp3lame CONFIG REQUIRED)

# 将源代码添加到此项目的可执行文件。
add_executable (wm_project    "wavelet.h" "main.cpp" "embed.cpp" "embed.h" "util.cpp" "extract.cpp" "wavelet.cpp" "global.h" "global.cpp" "new_embed.cpp" "new_embed.h" "dr_wav.h" "dr_wav.c" "tests/common/dr_common.h" "Log.cpp" "Log.h" "miniaudio.c"  "new_extract.cpp" "new_extract.h" )

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET wm_project PROPERTY CXX_STANDARD 17)
endif()

target_link_libraries(wm_project PRIVATE fmt::fmt FFTW3::fftw3l FFTW3::fftw3f FFTW3::fftw3 ${OpenCV_DIR} ${OpenCV_LIBS} SndFile::sndfile Eigen3::Eigen MPG123::libmpg123 MPG123::libout123 MPG123::libsyn123 mp3lame::mp3lame soxr)

