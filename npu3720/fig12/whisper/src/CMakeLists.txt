# TODO: should not use this
if (WIN32)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

if (WHISPER_COREML)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(COREML_FRAMEWORK CoreML)

    if (COREML_FRAMEWORK)
        message(STATUS "CoreML framework found")

        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_COREML)
    else()
        message(FATAL_ERROR "CoreML framework not found")
    endif()

    if (WHISPER_COREML_ALLOW_FALLBACK)
        set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_COREML_ALLOW_FALLBACK)
    endif()
endif()

if (WHISPER_OPENVINO)
    set(OpenVINO_DIR "/opt/intel/openvino_2024.4.0/runtime/cmake")
    find_package(OpenVINO REQUIRED COMPONENTS Runtime)
endif()

#
# libraries
#

# whisper.coreml

if (WHISPER_COREML)
    set(TARGET whisper.coreml)

    add_library(${TARGET}
        coreml/whisper-encoder.h
        coreml/whisper-encoder.mm
        coreml/whisper-encoder-impl.h
        coreml/whisper-encoder-impl.m
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC
        .
        )

    target_link_libraries(${TARGET} PRIVATE ${FOUNDATION_FRAMEWORK} ${COREML_FRAMEWORK})

    set_target_properties(${TARGET} PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
        XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES
        )
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

if (WHISPER_OPENVINO)
    set(TARGET whisper.openvino)

    add_library(${TARGET} OBJECT
        openvino/whisper-openvino-encoder.h
        openvino/whisper-openvino-encoder.cpp
        )

    target_include_directories(${TARGET} PUBLIC
        .
        )

    set_property(TARGET ${TARGET} PROPERTY POSITION_INDEPENDENT_CODE ON)
    set(WHISPER_EXTRA_FLAGS ${WHISPER_EXTRA_FLAGS} -DWHISPER_USE_OPENVINO)

    target_link_directories(${TARGET} PRIVATE )

    target_link_libraries(${TARGET} PRIVATE ggml openvino::runtime)

    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

#if (GGML_CUDA)
#    cmake_minimum_required(VERSION 3.18)  # for CMAKE_CUDA_ARCHITECTURES
#
#    find_package(CUDAToolkit)
#    if (CUDAToolkit_FOUND)
#        message(STATUS "CUDA found")
#
#        if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
#            # 52 == lowest CUDA 12 standard
#            # 60 == f16 CUDA intrinsics
#            # 61 == integer CUDA intrinsics
#            # 70 == compute capability at which unrolling a loop in mul_mat_q kernels is faster
#            set(CMAKE_CUDA_ARCHITECTURES "52;61;70") # lowest CUDA 12 standard + lowest for integer intrinsics
#        endif()
#        message(STATUS "Using CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
#
#        enable_language(CUDA)
#    else()
#        message(WARNING "CUDA not found")
#    endif()
#endif()

# whisper

add_library(whisper
            ../include/whisper.h
            whisper.cpp
            whisper-mel.hpp
            )

# TODO: disabled because it relies on ggml internals that are no longer accessible (ggml-backend-impl.h, ggml-cuda/common.cuh, ..)
#if (GGML_CUDA)
#    target_sources(whisper PRIVATE whisper-mel-cuda.cu)
#
#    target_link_libraries(whisper PRIVATE CUDA::cufft)
#endif()

# Set the version numbers
set_target_properties(whisper PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${SOVERSION}
)

target_include_directories(whisper PUBLIC . ../include)
target_compile_features   (whisper PUBLIC cxx_std_11) # don't bump

if (WHISPER_EXTRA_FLAGS)
    target_compile_options(whisper PRIVATE ${WHISPER_EXTRA_FLAGS})
endif()

target_link_libraries(whisper PUBLIC ggml)

if (WHISPER_COREML)
    target_link_libraries(whisper PRIVATE whisper.coreml)
endif()

if (WHISPER_OPENVINO)
    target_link_libraries(whisper PRIVATE whisper.openvino)
endif()

if (WHISPER_MKL)
    target_link_libraries(whisper PRIVATE MKL::MKL)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(whisper PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(whisper PRIVATE WHISPER_SHARED WHISPER_BUILD)
endif()
