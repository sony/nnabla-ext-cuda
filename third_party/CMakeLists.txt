function(download_and_extract_third_party_library name url)
  if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/${name})
    message("Downloading ${name}")
    file(DOWNLOAD
      ${url}
      ${CMAKE_SOURCE_DIR}/third_party/${name}.zip
      SHOW_PROGRESS
      STATUS status
      LOG log)
    list(GET status 0 status_code)
    list(GET status 1 status_string)
    if(NOT status_code EQUAL 0)
      if(UNIX OR APPLE)
        message("Failed to download ${name}. Try to download with wget.")
        execute_process(
          COMMAND wget ${url} -O ${CMAKE_SOURCE_DIR}/third_party/${name}.zip)
      else()
        message(FATAL_ERROR "Failed to download ${name}. Please check network connection and proxy settings.
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
      ${CMAKE_COMMAND} -E tar xf ${name}.zip)
  endif()
endfunction()

if(BUILD_TEST)
  download_and_extract_third_party_library(googletest-release-1.7.0 https://github.com/google/googletest/archive/release-1.7.0.zip)
  add_subdirectory(googletest-release-1.7.0)
endif()

download_and_extract_third_party_library(eigen-git-mirror-3.3.5 https://github.com/eigenteam/eigen-git-mirror/archive/3.3.5.zip)
