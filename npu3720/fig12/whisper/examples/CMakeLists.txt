# dependencies

find_package(Threads REQUIRED)

# third-party

if (WHISPER_SDL2)
    # SDL2
    find_package(SDL2 REQUIRED)

    string(STRIP "${SDL2_LIBRARIES}" SDL2_LIBRARIES)

    message(STATUS "SDL2_INCLUDE_DIRS = ${SDL2_INCLUDE_DIRS}")
    message(STATUS "SDL2_LIBRARIES    = ${SDL2_LIBRARIES}")
endif()

if (WHISPER_CLBLAST)
    find_package(CLBlast REQUIRED)
endif()

# common

set(TARGET common)

unset(COMMON_EXTRA_LIBS)

if (WHISPER_FFMPEG)
    # As of cmake 3.27, there is no official cmake support for FindFFmpeg.
    # Consequnelty we added a FindFFmpeg.cmake script the cmake subfolder:
    # whisper.cpp does not need the full ffmpeg libs, just AVFORMAT AVCODEC AVUTIL SWRESAMPLE
    # libswresample  performs highly optimized audio resampling, rematrixing and sample format conversion operations
    # libavcodec provides a generic encoding/decoding framework and contains multiple decoders and encoders for audio, video and subtitle streams, and several bitstream filters.
    # libavformat provides a generic framework for multiplexing and demultiplexing (muxing and demuxing) audio, video and subtitle streams.
    find_package(FFmpeg REQUIRED)

    if (NOT ${FFMPEG_FOUND})
        message(FATAL_ERROR "Cannot find ffmpeg libs/headers")
    endif()

    message(STATUS "Found ffmpeg libs:       ${FFMPEG_LIBRARIES}")
    message(STATUS "Found ffmpeg headers in: ${FFMPEG_INCLUDE_DIRS}")
    message(STATUS "ffmpeg definitions:      ${FFMPEG_DEFINITIONS}")
    message(STATUS "Found avformat           ${AVFORMAT_VERSION}")

    include_directories(${FFMPEG_INCLUDE_DIRS})
    add_compile_definitions(WHISPER_FFMPEG)

    list(APPEND COMMON_EXTRA_LIBS ${FFMPEG_LIBRARIES})

    set(COMMON_SOURCES_FFMPEG ffmpeg-transcode.cpp)
endif()


add_library(${TARGET} STATIC
    common.h
    common.cpp
    common-ggml.h
    common-ggml.cpp
    grammar-parser.h
    grammar-parser.cpp
    ${COMMON_SOURCES_FFMPEG}
    )

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE whisper ${COMMON_EXTRA_LIBS})

set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${TARGET} PROPERTIES FOLDER "libs")

if (WHISPER_SDL2)
    # common-sdl

    set(TARGET common-sdl)

    add_library(${TARGET} STATIC
        common-sdl.h
        common-sdl.cpp
        )

    include(DefaultTargetOptions)

    target_include_directories(${TARGET} PUBLIC  ${SDL2_INCLUDE_DIRS})
    target_link_libraries     (${TARGET} PRIVATE ${SDL2_LIBRARIES})

    set_target_properties(${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    set_target_properties(${TARGET} PROPERTIES FOLDER "libs")
endif()

# add json lib
add_library(json_cpp INTERFACE)
target_include_directories(json_cpp INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

# examples

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
    add_subdirectory(whisper.wasm)
    set_target_properties(libmain PROPERTIES FOLDER "libs")
    add_subdirectory(stream.wasm)
    set_target_properties(libstream PROPERTIES FOLDER "libs")
    add_subdirectory(command.wasm)
    set_target_properties(libcommand PROPERTIES FOLDER "libs")
    add_subdirectory(talk.wasm)
    set_target_properties(libtalk PROPERTIES FOLDER "libs")
    add_subdirectory(bench.wasm)
    set_target_properties(libbench PROPERTIES FOLDER "libs")
elseif(CMAKE_JS_VERSION)
    add_subdirectory(addon.node)
    set_target_properties(addon.node PROPERTIES FOLDER "examples")
else()
    # add_subdirectory(main)
    # set_target_properties(main PROPERTIES FOLDER "examples")
if (WHISPER_SDL2)
    add_subdirectory(stream)
    set_target_properties(stream PROPERTIES FOLDER "examples")
endif (WHISPER_SDL2)
#     add_subdirectory(server)
#     set_target_properties(server PROPERTIES FOLDER "examples")
# if (WHISPER_SDL2)
#     add_subdirectory(command)
#     set_target_properties(command PROPERTIES FOLDER "examples")
# endif (WHISPER_SDL2)
#     add_subdirectory(bench)
#     set_target_properties(bench PROPERTIES FOLDER "examples")
#     add_subdirectory(quantize)
#     set_target_properties(quantize PROPERTIES FOLDER "examples")
# if (WHISPER_SDL2)
#     add_subdirectory(talk)
#     set_target_properties(talk PROPERTIES FOLDER "examples")
#     add_subdirectory(talk-llama)
#     set_target_properties(talk-llama PROPERTIES FOLDER "examples")
#     add_subdirectory(lsp)
#     set_target_properties(lsp PROPERTIES FOLDER "examples")
#     if (GGML_SYCL)
#         add_subdirectory(sycl)
#         set_target_properties(sycl PROPERTIES FOLDER "examples")
#     endif()
# endif (WHISPER_SDL2)
endif()

# if (WHISPER_SDL2)
#     add_subdirectory(wchess)
#     set_target_properties(wchess PROPERTIES FOLDER "examples")
# endif (WHISPER_SDL2)
