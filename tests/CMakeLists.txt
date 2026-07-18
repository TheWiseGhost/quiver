find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(quiver_tests
  test_distance.cc
  test_thread_pool.cc
  test_flat.cc
  test_hnsw.cc
  test_persistence.cc
  test_quantizer.cc
  test_regressions.cc)
target_include_directories(quiver_tests PRIVATE ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(quiver_tests PRIVATE quiver quiver_options GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(quiver_tests DISCOVERY_TIMEOUT 60)
