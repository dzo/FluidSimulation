# Sets the minimum version of CMake required to build your native library.
# This ensures that a certain set of CMake features is available to
# your build.

cmake_minimum_required(VERSION 3.22)
get_filename_component(srcdir ${CMAKE_CURRENT_LIST_DIR} ABSOLUTE)
set(CMAKE_VERBOSE_MAKEFILE on)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -ftree-vectorize -ffast-math")
add_library(navierstokes SHARED jnisimulation.cpp )
if(${ANDROID_ABI} STREQUAL "arm64-v8a")
    add_library(navierstokes_gl SHARED jnisimulation.cpp )
    add_library(navierstokes_cl SHARED jnisimulation.cpp )
    add_library(navierstokes_vk SHARED jnisimulation.cpp )
    target_link_libraries(navierstokes_gl  jnigraphics GLESv3 EGL log ${srcdir}/${ANDROID_ABI}_gl/libnavierstokes_halide.a)
    target_link_libraries(navierstokes_cl  jnigraphics GLESv3 EGL log ${srcdir}/${ANDROID_ABI}_cl/libnavierstokes_halide.a)
    target_link_libraries(navierstokes_vk  jnigraphics GLESv3 EGL log vulkan ${srcdir}/${ANDROID_ABI}_vk/libnavierstokes_halide.a)
    target_include_directories(navierstokes_gl PUBLIC /usr/local/include/)
    target_include_directories(navierstokes_cl PUBLIC /usr/local/include/)
    target_include_directories(navierstokes_vk PUBLIC /usr/local/include/)
endif()
if(${ANDROID_ABI} STREQUAL "x86_64")
    add_library(navierstokes_gl SHARED jnisimulation.cpp )
    target_link_libraries(navierstokes_gl  jnigraphics GLESv3 EGL log ${srcdir}/${ANDROID_ABI}_gl/libnavierstokes_halide.a)
    target_include_directories(navierstokes_gl PUBLIC /usr/local/include/)
    add_library(navierstokes_vk SHARED jnisimulation.cpp )
    target_link_libraries(navierstokes_vk  jnigraphics GLESv3 EGL vulkan log ${srcdir}/${ANDROID_ABI}_vk/libnavierstokes_halide.a)
    target_include_directories(navierstokes_vk PUBLIC /usr/local/include/)
endif()

target_link_libraries(navierstokes  jnigraphics log ${srcdir}/${ANDROID_ABI}/libnavierstokes_halide.a)
target_include_directories(navierstokes PUBLIC /usr/local/include/)
#set(CMAKE_CXX_STANDARD 17)  # or newer
#set(CMAKE_CXX_STANDARD_REQUIRED YES)
#set(CMAKE_CXX_EXTENSIONS NO)
#find_package(Halide)
#add_executable(halide_generator IMPORTED)
#add_halide_library(halide_dens_step FROM halide_generator)
#add_halide_library(halide_vel_step FROM halide_generator)
#add_halide_library(halide_bitmap FROM halide_generator)



