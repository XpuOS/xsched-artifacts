project(Euler3D)

set(CMAKE_BUILD_TYPE Release)


cmake_minimum_required(VERSION 3.10)

enable_language(HIP)

add_executable(hip_euler ${CMAKE_CURRENT_LIST_DIR}/src/euler3d.hip)

target_compile_options(hip_euler PRIVATE -D__DRIVER_TYPES_H__ -D__CUDA_RUNTIME_H__)

target_include_directories(hip_euler PRIVATE
    ${HIP_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(hip_euler
    ${HIP_LIBRARIES}
)

install(TARGETS hip_euler RUNTIME DESTINATION bin)
