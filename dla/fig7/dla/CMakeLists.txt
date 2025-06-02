find_package(CUDA REQUIRED)

set(CUDA_SEARCH_PATHS PATHS "/usr" "/usr/local" "/usr/local/cuda")
set(CUDA_SEARCH_PATH_SUFFIXES lib lib64)

find_library(CUDLA_LIBRARY NAMES cudla ${CUDA_SEARCH_PATHS} PATH_SUFFIXES ${CUDA_SEARCH_PATH_SUFFIXES})
find_library(CUDART_LIBRARY NAMES cudart ${CUDA_SEARCH_PATHS} PATH_SUFFIXES ${CUDA_SEARCH_PATH_SUFFIXES})

file(GLOB_RECURSE DLA_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(dla_infer STATIC ${DLA_SRCS})

target_include_directories(dla_infer PUBLIC
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(dla_infer PUBLIC
    ${CUDLA_LIBRARY}
    ${CUDART_LIBRARY}
    XSched::utils
)
