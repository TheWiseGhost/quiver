cmake_minimum_required(VERSION 3.20)
project(quiver VERSION 0.1.0 DESCRIPTION "SIMD-accelerated approximate nearest neighbor search" LANGUAGES CXX)

option(QUIVER_BUILD_TESTS "Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(QUIVER_BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(QUIVER_BUILD_PYTHON "Build the Python extension module" OFF)
option(QUIVER_BUILD_FUZZERS "Build libFuzzer targets (needs a clang with -fsanitize=fuzzer)" OFF)
option(QUIVER_NATIVE "Tune for the build machine's CPU (-mcpu/-march=native)" OFF)
option(QUIVER_WERROR "Treat compiler warnings as errors" OFF)
set(QUIVER_SANITIZE "" CACHE STRING "Comma-separated sanitizers, e.g. address,undefined or thread")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  message(FATAL_ERROR "Quiver supports GCC and Clang (it relies on target attributes and __builtin_cpu_supports)")
endif()

# ---- shared compile settings -------------------------------------------------------------------
add_library(quiver_options INTERFACE)
target_compile_options(quiver_options INTERFACE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion -Wnon-virtual-dtor
  -Wold-style-cast -Woverloaded-virtual -Wnull-dereference -Wdouble-promotion
  $<$<BOOL:${QUIVER_WERROR}>:-Werror>)
if(QUIVER_NATIVE)
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    target_compile_options(quiver_options INTERFACE -mcpu=native)
  else()
    target_compile_options(quiver_options INTERFACE -march=native)
  endif()
endif()
if(QUIVER_SANITIZE)
  target_compile_options(quiver_options INTERFACE -fsanitize=${QUIVER_SANITIZE} -fno-omit-frame-pointer -fno-sanitize-recover=all)
  target_link_options(quiver_options INTERFACE -fsanitize=${QUIVER_SANITIZE})
endif()

# ---- library -----------------------------------------------------------------------------------
find_package(Threads REQUIRED)

add_library(quiver
  src/common.cc
  src/flat_index.cc
  src/hnsw_index.cc
  src/ivfpq_index.cc
  src/product_quantizer.cc
  src/scalar_quantizer.cc
  src/thread_pool.cc
  src/internal/chunked_storage.cc
  src/internal/crc32c.cc
  src/internal/file_format.cc
  src/simd/dispatch.cc
  src/simd/kernels_scalar.cc
  src/simd/kernels_neon.cc
  src/simd/kernels_avx2.cc)
add_library(quiver::quiver ALIAS quiver)
target_include_directories(quiver
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(quiver PUBLIC Threads::Threads PRIVATE $<BUILD_INTERFACE:quiver_options>)
target_compile_features(quiver PUBLIC cxx_std_20)

# ---- subprojects -------------------------------------------------------------------------------
if(QUIVER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
if(QUIVER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
if(QUIVER_BUILD_PYTHON)
  add_subdirectory(python)
endif()
if(QUIVER_BUILD_FUZZERS)
  add_subdirectory(fuzz)
endif()

# ---- install -----------------------------------------------------------------------------------
include(GNUInstallDirs)
install(TARGETS quiver EXPORT quiverTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/quiver DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT quiverTargets NAMESPACE quiver:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/quiver)
