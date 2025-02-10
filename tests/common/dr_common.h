//#include "dr_common.c"
#pragma once
#ifdef _WIN32
#include <windows.h>
#endif

#if defined(_MSC_VER) || defined(__DMC__)
#else
#include <strings.h>
#endif

#include <stdio.h>
#include <stdarg.h>
#include <time.h>   /* So we can seed the random number generator based on time. */
#include <errno.h>

#if !defined(_WIN32)
#include <dirent.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <unistd.h>
#include <dlfcn.h>
#endif

#include <stddef.h> /* For size_t. */

#ifdef _MSC_VER
#if defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wlanguage-extension-token"
#pragma GCC diagnostic ignored "-Wlong-long"
#pragma GCC diagnostic ignored "-Wc++11-long-long"
#endif
typedef   signed __int8  dr_int8;
typedef unsigned __int8  dr_uint8;
typedef   signed __int16 dr_int16;
typedef unsigned __int16 dr_uint16;
typedef   signed __int32 dr_int32;
typedef unsigned __int32 dr_uint32;
typedef   signed __int64 dr_int64;
typedef unsigned __int64 dr_uint64;
#if defined(__clang__)
#pragma GCC diagnostic pop
#endif
#else
#define MA_HAS_STDINT
#include <stdint.h>
typedef int8_t   dr_int8;
typedef uint8_t  dr_uint8;
typedef int16_t  dr_int16;
typedef uint16_t dr_uint16;
typedef int32_t  dr_int32;
typedef uint32_t dr_uint32;
typedef int64_t  dr_int64;
typedef uint64_t dr_uint64;
#endif

#ifdef MA_HAS_STDINT
typedef uintptr_t dr_uintptr;
#else
#if defined(_WIN32)
#if defined(_WIN64)
typedef dr_uint64 dr_uintptr;
#else
typedef dr_uint32 dr_uintptr;
#endif
#elif defined(__GNUC__)
#if defined(__LP64__)
typedef dr_uint64 dr_uintptr;
#else
typedef dr_uint32 dr_uintptr;
#endif
#else
typedef dr_uint64 dr_uintptr;   /* Fallback. */
#endif
#endif

typedef dr_uint8    dr_bool8;
typedef dr_uint32   dr_bool32;
#define DR_TRUE     1
#define DR_FALSE    0

typedef void* dr_handle;
typedef void* dr_ptr;
typedef void (*dr_proc)(void);

#if defined(SIZE_MAX)
#define DR_SIZE_MAX SIZE_MAX
#else
#define DR_SIZE_MAX 0xFFFFFFFF
#endif

int dr_strcpy_s(char* dst, size_t dstSizeInBytes, const char* src);

int dr_strncpy_s(char* dst, size_t dstSizeInBytes, const char* src, size_t count);

int dr_strcat_s(char* dst, size_t dstSizeInBytes, const char* src);

int dr_strncat_s(char* dst, size_t dstSizeInBytes, const char* src, size_t count);

int dr_append_path(char* dst, size_t dstSize, const char* base, const char* other);

const char* dr_path_file_name(const char* path);

const char* dr_extension(const char* path);

dr_bool32 dr_extension_equal(const char* path, const char* extension);

typedef struct
{
    char folderPath[256];
    char relativePath[256]; /* Relative to the original folder path. */
    char absolutePath[256]; /* Concatenation of folderPath and relativePath. */
    dr_bool32 isDirectory;
#ifdef _WIN32
    HANDLE hFind;
#else
    DIR* dir;
#endif
} dr_file_iterator;

dr_file_iterator* dr_file_iterator_begin(const char* pFolderPath, dr_file_iterator* pState);

dr_file_iterator* dr_file_iterator_next(dr_file_iterator* pState);

void dr_file_iterator_end(dr_file_iterator* pState);

static int dr_fopen(FILE** ppFile, const char* pFilePath, const char* pOpenMode);

void* dr_open_and_read_file_with_extra_data(const char* pFilePath, size_t* pFileSizeOut, size_t extraBytes);

void* dr_open_and_read_file(const char* pFilePath, size_t* pFileSizeOut);

dr_bool32 dr_argv_is_set(int argc, char** argv, const char* value);

int dr_vprintf_fixed(int width, const char* const format, va_list args);

int dr_printf_fixed(int width, const char* const format, ...);

int dr_vprintf_fixed_with_margin(int width, int margin, const char* const format, va_list args);

int dr_printf_fixed_with_margin(int width, int margin, const char* const format, ...);