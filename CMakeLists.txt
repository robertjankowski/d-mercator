cmake_minimum_required(VERSION 3.20)
project(mercator LANGUAGES Fortran CXX)

option(DMERCATOR_ENABLE_CUDA "Enable CUDA backend for Mercator refinement scoring" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wshadow -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -c -fpic -Wall -O3")

include_directories(include)
add_library(trapecis_Sd include/trapecis_Sd.f)
# add_library(trapecis_Sd_expected include/trapecis_Sd_expected.f)

set(source_dir "${PROJECT_SOURCE_DIR}/src/")
set(source_files "${source_dir}/embeddingSD_unix.cpp")

if(DMERCATOR_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "DMERCATOR_ENABLE_CUDA=ON but no CUDA compiler was found.")
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  list(APPEND source_files "${source_dir}/dmercator_cuda.cu")
endif()

set(include_dir "${PROJECT_SOURCE_DIR}/include/")
file(GLOB include_files "${include_dir}/*.hpp")

add_executable(mercator ${source_files} ${include_files})

find_package(Threads REQUIRED)

if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  find_path(LIBOMP_INCLUDE_DIR
    NAMES omp.h
    HINTS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include
  )
  find_library(LIBOMP_LIBRARY
    NAMES omp
    HINTS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib
  )

  if(NOT LIBOMP_INCLUDE_DIR OR NOT LIBOMP_LIBRARY)
    message(FATAL_ERROR
      "OpenMP runtime (libomp) not found for AppleClang. Install with: brew install libomp"
    )
  endif()

  set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_INCLUDE_DIR}")
  set(OpenMP_CXX_LIB_NAMES "omp")
  set(OpenMP_omp_LIBRARY "${LIBOMP_LIBRARY}")
endif()

find_package(OpenMP REQUIRED COMPONENTS CXX)

target_link_libraries(mercator PRIVATE trapecis_Sd Threads::Threads OpenMP::OpenMP_CXX)

if(DMERCATOR_ENABLE_CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_compile_definitions(mercator PRIVATE DMERCATOR_USE_CUDA=1)
  target_link_libraries(mercator PRIVATE CUDA::cudart)
  set_target_properties(mercator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
