set(ACL_INCLUDE_DIRS
    "/usr/local/Ascend/ascend-toolkit/latest/include"
)

set(ACL_LINK_DIRS
    "/usr/local/Ascend/driver/lib64/driver"
    "/usr/local/Ascend/ascend-toolkit/latest/lib64"
)

file(GLOB_RECURSE ACL_INFER_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/src/*.c
    ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp
)

add_library(acl_infer STATIC ${ACL_INFER_SRCS})

target_include_directories(acl_infer PUBLIC
    ${ACL_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_directories(acl_infer PUBLIC
    ${ACL_LINK_DIRS}
)

target_link_libraries(acl_infer PUBLIC
    pthread
    ascendcl
    XSched::utils
)

target_compile_options(acl_infer PRIVATE
    -Wall
    -Wextra
    -Werror
    -std=c++11
)
