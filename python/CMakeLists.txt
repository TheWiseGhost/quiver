find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_quiver NB_STATIC STABLE_ABI LTO quiver_ext.cc)
target_link_libraries(_quiver PRIVATE quiver)

if(SKBUILD)
  install(TARGETS _quiver LIBRARY DESTINATION quiver)
else()
  # In-tree development build: drop the extension next to the Python package sources.
  set_target_properties(_quiver PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/quiver)
endif()
