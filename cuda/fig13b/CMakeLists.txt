cmake_minimum_required(VERSION 3.22)
project(xsched-tvm)

set(BIGDISK_INSTALL_DIR "/bigdisk/install" CACHE PATH "Path to the installation directory")

set(CMAKE_INSTALL_RPATH "${BIGDISK_INSTALL_DIR}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

find_package(CUDAToolkit)

include_directories(${CUDAToolkit_INCLUDE_DIRS})

find_package(XSched REQUIRED)

add_executable(xsched_tvm src/main.cpp)

find_library(TVM_RUNTIME_LIB tvm_runtime PATHS ${BIGDISK_INSTALL_DIR}/lib REQUIRED)
include_directories(${BIGDISK_INSTALL_DIR}/include)

target_link_libraries(xsched_tvm 
    CUDA::cudart
    XSched::preempt
    XSched::halcuda
    ${TVM_RUNTIME_LIB}
)

