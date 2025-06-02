list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)

find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)

file(GLOB_RECURSE TRT_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cu
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(trt_infer STATIC ${TRT_SRCS})

target_include_directories(trt_infer PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(trt_infer PUBLIC
    ${CUDA_LIBRARIES}
    TensorRT::nvinfer
    TensorRT::nvinfer_plugin
    TensorRT::nvonnxparser
    XSched::preempt
    XSched::halcuda
    XSched::utils
)
