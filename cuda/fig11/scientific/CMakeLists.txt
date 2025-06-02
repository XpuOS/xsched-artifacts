find_package(CUDA REQUIRED)

enable_language(CUDA)

cuda_add_executable(cuda_euler ${CMAKE_CURRENT_LIST_DIR}/src/euler3d.cu)

set_target_properties(cuda_euler PROPERTIES CUDA_ARCHITECTURES "${CUDA_GEN_CODE}")

target_include_directories(cuda_euler PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(cuda_euler
    ${CUDA_LIBRARIES}
)

install(TARGETS cuda_euler RUNTIME DESTINATION bin)
