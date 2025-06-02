cmake_minimum_required(VERSION 3.12)

project(dla_fig7 VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS_RELEASE   "$ENV{CFLAGS}   -O3 -DRELEASE_MODE")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DRELEASE_MODE")
set(CMAKE_C_FLAGS_DEBUG     "$ENV{CFLAGS}   -O0 -g2 -ggdb -DDEBUG_MODE")
set(CMAKE_CXX_FLAGS_DEBUG   "$ENV{CXXFLAGS} -O0 -g2 -ggdb -DDEBUG_MODE")

add_subdirectory(dla)

set(PLATFORM_CUDLA ON CACHE BOOL "Build CUDLA support.")
add_subdirectory(${XSCHED_DIR} xsched)

file(GLOB_RECURSE MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)
foreach(MAIN_SRC ${MAIN_SRCS})
    get_filename_component(MAIN_TARGET ${MAIN_SRC} NAME_WE)
    add_executable(${MAIN_TARGET} ${MAIN_SRC})
    target_link_libraries(${MAIN_TARGET} PRIVATE
        dla_infer
        XSched::utils
    )
    install(TARGETS ${MAIN_TARGET} RUNTIME DESTINATION bin)
endforeach()
