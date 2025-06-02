find_package(vpi REQUIRED)
find_package(OpenCV REQUIRED)

file(GLOB_RECURSE VPI_RUNNER_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(vpi_runner STATIC ${VPI_RUNNER_SRCS})

target_include_directories(vpi_runner PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(vpi_runner PUBLIC
    vpi
    pthread
    opencv_core
    opencv_videoio
    opencv_imgproc
    opencv_imgcodecs
    XSched::utils
)
