find_package(benchmark CONFIG QUIET)
if(NOT benchmark_FOUND)
  include(FetchContent)
  FetchContent_Declare(benchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(benchmark)
endif()

add_executable(bench_kernels bench_kernels.cc)
target_link_libraries(bench_kernels PRIVATE quiver quiver_options benchmark::benchmark)

add_executable(ann_bench ann_bench.cc dataset.cc)
target_link_libraries(ann_bench PRIVATE quiver quiver_options)
