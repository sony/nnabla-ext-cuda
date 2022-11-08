function(download_and_extract_library NAME EXT URL)
  if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/${NAME})
    message("Downloading ${NAME}")
    file(DOWNLOAD
      ${URL}
      ${CMAKE_SOURCE_DIR}/third_party/${NAME}${EXT}
      SHOW_PROGRESS
      STATUS status
      LOG log)
    list(GET status 0 status_code)
    list(GET status 1 status_string)
    if(NOT status_code EQUAL 0)
      if(UNIX OR APPLE)
        message("Failed to download ${NAME}. Try to download with wget.")
        execute_process(
          COMMAND wget ${WGET_OPTS} ${URL} -O ${CMAKE_SOURCE_DIR}/third_party/${NAME}${EXT})
      else()
        message(FATAL_ERROR "Failed to download ${NAME}. Please check network connection and proxy settings.
        status_code: ${status_code}
        status_string: ${status_string}
        log: ${log}
        ")
        error() # Place Unknown command to stop cmake process here.
      endif()

    endif()
    execute_process(
      COMMAND ${CMAKE_COMMAND} -E
      chdir ${CMAKE_SOURCE_DIR}/third_party
      ${CMAKE_COMMAND} -E tar xf ${NAME}${EXT})
  endif()
endfunction()

if(BUILD_CPP_LIB)
  download_and_extract_library(eigen-3.3.5 .zip https://gitlab.com/libeigen/eigen/-/archive/3.3.5/eigen-3.3.5.zip)
  download_and_extract_library(ompi-3.1.6 .zip https://github.com/open-mpi/ompi/archive/refs/tags/v3.1.6.zip)
  # cub is included for CUDA Toolkit >= 11.0.
  if(CUDA_VERSION VERSION_LESS "11.0")
    download_and_extract_library(cub-1.8.0 .zip https://github.com/NVIDIA/cub/archive/refs/tags/1.8.0.zip)
  endif()

  if (WIN32)
  else ()
    # Generate dl_mpi.h from ompi.h.in
    execute_process(
      COMMAND ${PYTHON_COMMAND}
              ${CMAKE_SOURCE_DIR}/build-tools/code_generator/dl_mpi_generate.py
              OUTPUT_VARIABLE TEST
              RESULT_VARIABLE GENERATE_RESULT)
    if (NOT ${GENERATE_RESULT} EQUAL 0)
      message(FATAL_ERROR "mpi header generated failed.")
    endif()

    if(WITH_NCCL)
      # Generate dl_nccl.h from nccl.h
      execute_process(
        COMMAND ${PYTHON_COMMAND}
                ${CMAKE_SOURCE_DIR}/build-tools/code_generator/dl_nccl_generate.py
                OUTPUT_VARIABLE TEST
                RESULT_VARIABLE GENERATE_RESULT)
      if (NOT ${GENERATE_RESULT} EQUAL 0)
        message(FATAL_ERROR "nccl header and cpp file generated failed.")
      endif()
    endif()
  endif()

  if(BUILD_TEST)
    download_and_extract_library(googletest-release-1.10.0 .zip https://github.com/google/googletest/archive/release-1.10.0.zip)
    add_subdirectory(googletest-release-1.10.0)
  endif()
endif()

if(UNIX)
  set(cutensor_default_path /usr/lib64 /usr/lib/x86_64-linux-gnu)
  if(${CUDA_VERSION} VERSION_GREATER 11.0)
    set(cutensor_ver 11)
  else()
    set(cutensor_ver ${CUDA_VERSION})
  endif()
  foreach(path ${cutensor_default_path})
    if(EXISTS ${path}/libcutensor)
      set(cutensor_lib_path ${path}/libcutensor/${cutensor_ver} PARENT_SCOPE)
      set(cutensor_include_path /usr/include PARENT_SCOPE)
      set(cutensor_found TRUE)
    endif()
  endforeach()
  if(NOT (DEFINED cutensor_found))
    set(cutensor_file_name libcutensor-linux-x86_64-1.6.0.3-archive)
    if(NOT EXISTS ${CMAKE_SOURCE_DIR}/third_party/${cutensor_file_name})
      download_and_extract_library(${cutensor_file_name} .tar.xz https://developer.download.nvidia.com/compute/cutensor/redist/libcutensor/linux-x86_64/${cutensor_file_name}.tar.xz)
    endif()
    set(cutensor_lib_path ${CMAKE_SOURCE_DIR}/third_party/${cutensor_file_name}/lib/${cutensor_ver} PARENT_SCOPE)
    set(cutensor_include_path ${CMAKE_SOURCE_DIR}/third_party/${cutensor_file_name}/include PARENT_SCOPE)
  endif()
endif()
if(WIN32)
    set(cutensor_include_path ${CUTENSOR_INCLUDE_DIR} PARENT_SCOPE)
endif()
