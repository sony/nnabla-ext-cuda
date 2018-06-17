# Copyright (c) 2017 Sony Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required (VERSION 3.1)

# NNabla project
project (nbla C CXX)

###############################################################################
option(BUILD_CPP_LIB "Build C++ Library" ON)
option(BUILD_CPP_TEST "Build testing" OFF)
option(BUILD_PYTHON_PACKAGE "Build python package" ON)
option(MAKE_MANYLINUX_WHEEL "Convert wheel to manylinux version" OFF)

###############################################################################
# Settings
###############################################################################
if(NOT IS_DIRECTORY ${NNABLA_DIR})
  message(FATAL_ERROR "Invalid NNABLA_DIR :[${NNABLA_DIR}]")
endif()
if(NOT IS_ABSOLUTE ${NNABLA_DIR})
  get_filename_component(NNABLA_DIR ${CMAKE_CURRENT_BINARY_DIR}/${NNABLA_DIR} ABSOLUTE)
endif()

if(NOT EXISTS ${CPPLIB_LIBRARY})
  message(FATAL_ERROR "Invalid CPPLIB_LIBRARY :[${CPPLIB_LIBRARY}]")
endif()
if(NOT IS_ABSOLUTE ${CPPLIB_LIBRARY})
  get_filename_component(CPPLIB_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/${CPPLIB_LIBRARY} ABSOLUTE)
endif()

option(NBLA_VERBOSE_MEMORY_USAGE "Enable debug print of memory usage" OFF)
if(NBLA_VERBOSE_MEMORY_USAGE)
  add_definitions(-DNBLA_VERBOSE_MEMORY_USAGE)
endif()

if (NOT PYTHON_COMMAND_NAME)
  if(NOT PYTHON_VERSION_MAJOR)
    set(PYTHON_COMMAND_NAME "python")
  else()
    set(PYTHON_COMMAND_NAME "python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}")
  endif()
endif()

find_program(PYTHON_COMMAND ${PYTHON_COMMAND_NAME})
if(NOT EXISTS ${PYTHON_COMMAND})
  find_program(PYTHON_COMMAND "python")
  if(NOT EXISTS ${PYTHON_COMMAND})
    message(FATAL_ERROR, "python [${PYTHON_COMMAND_NAME}] not found. Please install python or check your settings.")
    error() # Place Unknown command to stop cmake process here.
  endif()
endif()

###############################################################################
# General settings
###############################################################################
# Using cmake scripts and modules
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/build-tools/cmake/Modules)
include(${NNABLA_DIR}/build-tools/cmake/Utils.cmake)

###############################################################################
# Code generator
###############################################################################
if (MSVC)
  set(ENV{PYTHONPATH} ${NNABLA_DIR}\\build-tools\\code_generator;$ENV{PYTHONPATH})
else()
  set(ENV{PYTHONPATH} ${NNABLA_DIR}/build-tools/code_generator:$ENV{PYTHONPATH})
endif()

execute_process(
  COMMAND ${PYTHON_COMMAND}
          ${CMAKE_CURRENT_SOURCE_DIR}/build-tools/code_generator/generate.py
          OUTPUT_VARIABLE TEST
          RESULT_VARIABLE GENERATE_RESULT)
if (NOT ${GENERATE_RESULT} EQUAL 0)
  message(FATAL_ERROR "generate.py failed.")
endif()

if(BUILD_CPP_LIB)
  ###############################################################################
  # Third party libraries
  ###############################################################################
  add_subdirectory(third_party)
  
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type release is default on single-configuration build system like GnuMake."
      FORCE)
  endif()
  
  # Flags
  if(MSVC)
    # Force to always compile with W4
    if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
      string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
    endif()
  endif()
  
  if(UNIX OR APPLE)
    find_program(CCACHE_FOUND ccache)
    if(CCACHE_FOUND)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
      set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    endif(CCACHE_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wall")
  endif()
  nbla_warnings_disable(CMAKE_CXX_FLAGS -Wno-sign-compare -Wno-uninitialized /wd4099)
  
  # Setting output directory naively
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  if (WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  endif()
  
  ###############################################################################
  # C++ core
  ###############################################################################
  # includes
  list(APPEND NBLA_INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${NNABLA_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen-eigen-de14376a69f8)
  include_directories(${NBLA_INCLUDE_DIRS};${PROJECT_BINARY_DIR})
  
  ###############################################################################
  # C++ CUDA core
  ###############################################################################
  add_definitions(-DWITH_CUDA)
  add_definitions(-DWITH_CUDNN)
  add_subdirectory(src/nbla/cuda)

  ###############################################################################
  # C++ Testing
  ###############################################################################
  if(BUILD_CPP_TEST)
    if (WIN32)
      set(BUILD_SHARED_LIBS ON CACHE BOOL
        "Force to build gtest with dll on Windows.")
    endif()
    enable_testing()
    include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
    include_directories(${NBLA_CUDA_INCLUDE_DIRS})
  
    file(GLOB NBLA_TEST_SOURCES src/nbla/cuda/test/test_*.cpp)
    add_executable(clibtest ${NBLA_TEST_SOURCES})
    add_dependencies(clibtest ${NBLA_LIBRARY_NAME} ${NBLA_CUDA_LIBRARY_NAME})
    target_link_libraries(clibtest gtest gtest_main)
    target_link_libraries(clibtest nnabla)
  
    add_dependencies(clibtest ${NBLA_CUDA_LIBRARY_NAME})
    target_link_libraries(clibtest ${NBLA_CUDA_LIBRARY_NAME})
    target_link_libraries(clibtest ${NBLA_CUDA_LINKER_LIBS})
  
    set_property(TARGET clibtest PROPERTY CXX_STANDARD 11)
    nbla_exclude_from_all(clibtest)  # Exclude target from all or default build
    add_test(NAME clibtest COMMAND clibtest)
  endif()

  ###############################################################################
  # Generate setup.cfg
  ###############################################################################
  if (NOT (DEFINED NBLA_PYTHON_BUILD_EXT_COMPILER))
    if (WIN32)
      set(NBLA_PYTHON_BUILD_EXT_COMPILER msvc)
    elseif(UNIX OR APPLE)
      set(NBLA_PYTHON_BUILD_EXT_COMPILER unix)
    else ()
      message(FATAL_ERROR
        "Python build_ext compiler inference is only supported on Win, Unix or Apple.")
    endif()
    message("Python build_ext compiler is infered as '${NBLA_PYTHON_BUILD_EXT_COMPILER}'.")
    message("You can specify a compiler manually setting a variable"
      " NBLA_PYTHON_BUILD_EXT_COMPILER. You can see a list of supported"
      " compiler by `python setup.py build_ext --help-compiler`.")
  endif()
  set(NBLA_PYTHON_BUILD_DIR ${CMAKE_BINARY_DIR})

  # Two-step configure_file generation with generate-time property (target_file_name).
  # http://cmake.3232098.n2.nabble.com/cmp0026-file-GENERATE-and-configure-file-td7590275.html
  # Since multiple-configuration build system like Visual Studio requires
  # file(GENERATE ...) for each build config, we generate setup.cfg files for each
  # build type with suffix `-<config>`. You can switch the build type by replacing
  # the setup.cfg file by setup.cfg-<config>. The default setup.cfg is for Release build.
  configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in
    ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in.configured)
  
  file(GENERATE OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg-$<CONFIG>
    INPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in.configured)
  
  file(GENERATE OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg
    INPUT ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.in.configured
    CONDITION $<CONFIG:Release>)
  
  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg
    ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg-$<CONFIG>)

endif() # BUILD_CPP_LIB

###############################################################################
# Python API
###############################################################################
if(BUILD_PYTHON_PACKAGE)

  if(NOT BUILD_CPP_LIB)
    if(NOT IS_DIRECTORY ${CPPLIB_BUILD_DIR})
      message(FATAL_ERROR "Invalid CPPLIB_BUILD_DIR :[${CPPLIB_BUILD_DIR}]")
    endif()
    if(NOT EXISTS ${CPPLIB_LIBRARY})
      message(FATAL_ERROR "Invalid CPPLIB_LIBRARY :[${CPPLIB_LIBRARY}]")
    endif()
    if(NOT EXISTS ${CPPLIB_CUDA_LIBRARY})
      message(FATAL_ERROR "Invalid CPPLIB_LIBRARY :[${CPPLIB_CUDA_LIBRARY}]")
    endif()

    # No one sets the following two vars if !BUILD_CPP_LIB
    set(NBLA_LIBRARY_NAME ${CPPLIB_LIBRARY})
    set(NBLA_CUDA_LIBRARY_NAME ${CPPLIB_CUDA_LIBRARY})

    set(CPPLIB_LIBRARY_NAME "nnabla")
    set(CPPLIB_TARGET_FILE ${CPPLIB_LIBRARY})
    get_filename_component(CPPLIB_TARGET_FILE_NAME ${CPPLIB_TARGET_FILE} NAME)

    set(CPPLIB_CUDA_LIBRARY_NAME "nnabla_cuda")
    set(CPPLIB_CUDA_TARGET_FILE ${CPPLIB_CUDA_LIBRARY})
    get_filename_component(CPPLIB_CUDA_TARGET_FILE_NAME ${CPPLIB_CUDA_TARGET_FILE} NAME)
    
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg.build-wheel.in
      ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.cfg)
  endif()


  set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "build")
  # http://stackoverflow.com/questions/13298504/using-cmake-with-setup-py
  
  file(GLOB_RECURSE NBLA_PYTHON_DEPS
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.py
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.pyx
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.pxd)
  
  file(GLOB_RECURSE NBLA_PYTHON_BUILD_OUTPUT_CYTHON
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/python/src/*.h)
  
  set(NBLA_PYTHON_SETUP ${CMAKE_CURRENT_SOURCE_DIR}/python/setup.py)
  set(NBLA_PYTHON_OUTPUT ${CMAKE_BINARY_DIR}/build)
  set(NBLA_PYTHON_OUTPUT_WHEEL ${NBLA_PYTHON_OUTPUT}/.timestamp.wheel)
  set(NBLA_WHEEL_DEPENDS ${NBLA_PYTHON_DEPS} ${NBLA_CUDA_LIBRARY_NAME})
  
  # Wheel
  if(UNIX)
    if(MAKE_MANYLINUX_WHEEL)
      add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_WHEEL}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*.whl
        COMMAND ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} -q bdist_wheel
        COMMAND ${NNABLA_DIR}/build-tools/auditwheel-nnabla ${CMAKE_BINARY_DIR}/dist/*-linux_*.whl
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*-linux*.whl
        COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_WHEEL}
        DEPENDS ${NBLA_WHEEL_DEPENDS})
    else()
      add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_WHEEL}
        COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/dist/*.whl
        COMMAND ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} -q bdist_wheel
        COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_WHEEL}
        DEPENDS ${NBLA_WHEEL_DEPENDS})
    endif()
  else()
    add_custom_command(OUTPUT ${NBLA_PYTHON_OUTPUT_WHEEL}
      COMMAND ${PYTHON_COMMAND} ${NBLA_PYTHON_SETUP} -q bdist_wheel
      COMMAND ${CMAKE_COMMAND} -E touch ${NBLA_PYTHON_OUTPUT_WHEEL}
      DEPENDS ${NBLA_WHEEL_DEPENDS})
  endif()

  if(BUILD_CPP_LIB)
    add_custom_target(wheel ALL DEPENDS ${NBLA_PYTHON_OUTPUT_WHEEL})
  else()
    add_custom_target(wheel DEPENDS ${NBLA_PYTHON_OUTPUT_WHEEL})
  endif()

endif() # BUILD_PYTHON_PACKAGE
