project(BlackScholes)

cmake_minimum_required(VERSION 3.10)


# Find HIP package
find_package(hip REQUIRED)

enable_language(HIP)

add_executable(hip_bs ${CMAKE_CURRENT_LIST_DIR}/src/BlackScholes.hip)

target_compile_options(hip_bs PRIVATE -D__DRIVER_TYPES_H__ -D__CUDA_RUNTIME_H__)

target_include_directories(hip_bs PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
    ${HIP_INCLUDE_DIRS}
    ${hip_INCLUDE_DIRS}
)

target_link_libraries(hip_bs
    ${HIP_LIBRARIES}
    hip::host
)

install(TARGETS hip_bs RUNTIME DESTINATION bin)
