# Sets the minimum version of CMake required to build your native library.
# This ensures that a certain set of CMake features is available to
# your build.

cmake_minimum_required(VERSION 3.4.1)
get_filename_component(srcdir ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -ftree-vectorize -ffast-math")
add_library(navierstokes SHARED jnisimulation.cpp )
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    add_library(navierstokes_gl SHARED jnisimulation.cpp )
    add_library(navierstokes_cl SHARED jnisimulation.cpp )
    target_link_libraries(navierstokes_gl  jnigraphics GLESv3 EGL log ${srcdir}/${ANDROID_ABI}_gl/libnavierstokes_halide.a)
    target_link_libraries(navierstokes_cl  jnigraphics GLESv3 EGL log ${srcdir}/${ANDROID_ABI}_cl/libnavierstokes_halide.a)
    target_include_directories(navierstokes_gl PUBLIC /usr/local/include/)
    target_include_directories(navierstokes_cl PUBLIC /usr/local/include/)
endif()
if(${ANDROID_ABI} STREQUAL "x86_64")
    add_library(navierstokes_gl SHARED jnisimulation.cpp )
    target_link_libraries(navierstokes_gl  jnigraphics GLESv3 EGL log ${srcdir}/${ANDROID_ABI}_gl/libnavierstokes_halide.a)
    target_include_directories(navierstokes_gl PUBLIC /usr/local/include/)
endif()

target_link_libraries(navierstokes  jnigraphics log ${srcdir}/${ANDROID_ABI}/libnavierstokes_halide.a)
target_include_directories(navierstokes PUBLIC /usr/local/include/)


