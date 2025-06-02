cmake_minimum_required(VERSION 3.12)

project(ascend_fig7 VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS_RELEASE   "$ENV{CFLAGS}   -O3 -DRELEASE_MODE")
set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -DRELEASE_MODE")
set(CMAKE_C_FLAGS_DEBUG     "$ENV{CFLAGS}   -O0 -g2 -ggdb -DDEBUG_MODE")
set(CMAKE_CXX_FLAGS_DEBUG   "$ENV{CXXFLAGS} -O0 -g2 -ggdb -DDEBUG_MODE")

set(PLATFORM_ASCEND ON CACHE BOOL "Build ASCEND support.")
add_subdirectory(${XSCHED_DIR} xsched)
add_subdirectory(acl)

file(GLOB_RECURSE MAIN_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/main/*.c
    ${CMAKE_CURRENT_LIST_DIR}/main/*.cpp
)

foreach(MAIN_SRC ${MAIN_SRCS})
    get_filename_component(MAIN_TARGET ${MAIN_SRC} NAME_WE)
    add_executable(${MAIN_TARGET} ${MAIN_SRC})
    target_link_libraries(${MAIN_TARGET}
        acl_infer
        XSched::utils
    )
    set_target_properties(${MAIN_TARGET} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH_USE_LINK_PATH TRUE
    )
    install(TARGETS ${MAIN_TARGET} RUNTIME DESTINATION bin)
endforeach()
