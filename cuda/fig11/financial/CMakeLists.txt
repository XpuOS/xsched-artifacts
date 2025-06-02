find_package(CUDA REQUIRED)

enable_language(CUDA)

cuda_add_executable(cuda_bs ${CMAKE_CURRENT_LIST_DIR}/src/BlackScholes.cu)

set_target_properties(cuda_bs PROPERTIES CUDA_ARCHITECTURES "${CUDA_GEN_CODE}")

target_include_directories(cuda_bs PRIVATE
    ${CUDA_INCLUDE_DIRS}
    ${CMAKE_CURRENT_LIST_DIR}/include
)

target_link_libraries(cuda_bs
    ${CUDA_LIBRARIES}
)
install(TARGETS cuda_bs RUNTIME DESTINATION bin)
