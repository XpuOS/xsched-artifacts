set(OpenVINO_DIR "/opt/intel/openvino_2024.4.0/runtime/cmake")
find_package(OpenVINO REQUIRED)

file(GLOB_RECURSE OV_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(ov_infer STATIC ${OV_SRCS})

target_include_directories(ov_infer PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(ov_infer PUBLIC
    openvino::runtime
    XSched::utils
)
