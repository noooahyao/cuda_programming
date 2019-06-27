project(sdnn LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.8)

#list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#set(BLA_DEFINITIONS)
#if ("$ENV{MKLROOT}" STREQUAL "")
#    message(FATAL_ERROR "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
#endif()
## user defined options for MKL
#option(MKL_USE_parallel "Use MKL parallel" True)
#option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
#set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")
#if (BLA_VENDOR MATCHES "_seq")
#    set(MKL_USE_parallel OFF)
#else()
#    set(MKL_USE_parallel ON)
#endif()
#find_package(MKL REQUIRED)
#include_directories(${MKL_INCLUDE_DIR})
#if (MKL_FOUND)
#    set(BLA_INCLUDE_DIR ${MKL_INCLUDE_DIR})
#    set(BLAS_LIBRARIES ${MKL_LIBRARIES})
#    set(BLA_DEFINITIONS USE_MKL)
#endif()
#message(STATUS "BLAS Libraries: ${BLAS_LIBRARIES}")




add_compile_options(-std=c++11)


#set_property(GLOBAL PROPERTY CUDA_STANDARD 11)
############## DEPENDENCIES ##############
#find_package(CUDA REQUIRED)
#set(CUDA_SEPARABLE_COMPILATION ON)
#message("-- CUDA Version: ${CUDA_VERSION_STRING}")

include(CheckLanguage)
check_language(CUDA)

# GFlags
add_subdirectory(deps/gflags)
include_directories(${PROJECT_BINARY_DIR}/deps/gflags/include)
link_directories(${PROJECT_BINARY_DIR}/deps/gflags/lib)


# cub
#add_subdirectory(deps/cub)
include_directories(deps/cub)


############## BUILD ##############  -gencode arch=compute_60,code=sm_60 arch=compute_75,code=sm_75
set(EXTRA_LIBS gflags)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}  -g -G -I./deps/cub -lineinfo -lcusparse  -arch=sm_75  -Xcompiler -ggdb ")
else()
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -I./deps/cub -lcusparse -gencode -arch=sm_75  -O3 -DNDEBUG -Xcompiler -DNDEBUG ")
endif()

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lcusparse  -std=c++11 -DNDEBUG --expt-extended-lambda -Xptxas="-v")

#if(NOT DEFINED CMAKE_CUDA_STANDARD)
#    set(CMAKE_CUDA_STANDARD 11)
#    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#endif()

#source directory
AUX_SOURCE_DIRECTORY(src/include DIR_SRCS)
include_directories(src/include)

add_executable(sdnn
    src/sdnn.cu
    ${DIR_SRCS})
target_link_libraries(sdnn ${EXTRA_LIBS} -lcusparse )

add_executable(sdnn_filter
    src/sdnn_filter.cu
    ${DIR_SRCS})
target_link_libraries(sdnn_filter ${EXTRA_LIBS} -lcusparse )

add_executable(sdnn_stream
    src/sdnn_stream.cu
    ${DIR_SRCS})
target_link_libraries(sdnn_stream ${EXTRA_LIBS} -lcusparse )

add_executable(sdnn_cusparse
    src/sdnn_cusparse.cu 
    ${DIR_SRCS})
target_link_libraries(sdnn_cusparse ${EXTRA_LIBS} -lcusparse)

add_executable(sdnn_cusparse_nobuffer
    src/sdnn_cusparse_nobuffer.cu
    ${DIR_SRCS})
target_link_libraries(sdnn_cusparse_nobuffer ${EXTRA_LIBS} -lcusparse)
#add_executable(sdnn_mkl
#    src/include/ref_mkl.hpp)
#set_target_properties(sdnn_mkl PROPERTIES
#    CXX_STANDARD 11
#    CXX_STANDARD_REQUIRED ON
#    CXX_EXTENSIONS OFF
#)
#target_link_libraries(sdnn_mkl ${EXTRA_LIBS} ${MKL_LIBRARIES})




