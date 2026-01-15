cmake_minimum_required(VERSION 3.20)
project(mercator Fortran CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 -fopenmp -lpthread") # with -O3 compiler changes the code...
set(CMAKE_Fortran_FLAGS "-c -fpic -Wall -O3")

include_directories(include)
include_directories(include/*)
add_library(trapecis_Sd include/trapecis_Sd.f)
# add_library(trapecis_Sd_expected include/trapecis_Sd_expected.f)

set(source_dir "${PROJECT_SOURCE_DIR}/src/")
set(source_files "${source_dir}/embeddingSD_unix.cpp")

set(include_dir "${PROJECT_SOURCE_DIR}/include/")
file(GLOB include_files "${include_dir}/*.hpp")

add_executable(mercator ${source_files} ${include_files})
target_link_libraries(mercator trapecis_Sd)

option(MERCATOR_ENABLE_CUDA "Enable CUDA acceleration for loglikelihood calculations" ON)
if(MERCATOR_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    target_sources(mercator PRIVATE "${source_dir}/embeddingSD_cuda.cu")
    target_link_libraries(mercator PRIVATE CUDA::cudart)
    target_compile_definitions(mercator PRIVATE MERCATOR_USE_CUDA)
  endif()
endif()

enable_testing()
add_executable(loglikelihood_tests test/loglikelihood_cuda_test.cpp)
target_include_directories(loglikelihood_tests PRIVATE "${include_dir}")
if(CUDAToolkit_FOUND)
  target_sources(loglikelihood_tests PRIVATE "${source_dir}/embeddingSD_cuda.cu")
  target_link_libraries(loglikelihood_tests PRIVATE CUDA::cudart)
  target_compile_definitions(loglikelihood_tests PRIVATE MERCATOR_USE_CUDA)
endif()
add_test(NAME loglikelihood_tests COMMAND loglikelihood_tests)
