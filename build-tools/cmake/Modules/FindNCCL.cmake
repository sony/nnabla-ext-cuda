# Find NCCL, NVIDIA Collective Communications Library
# Returns:
#   NCCL_INCLUDE_DIR
#   NCCLLIBRARIES

find_path(NCCL_INCLUDE_DIR NAMES nccl.h 
		PATHS 
	  /usr/include
		/usr/local/include
		$ENV{NCCL_HOME}/include)
find_library(NCCL_LIBRARIES NAMES nccl 
		PATHS 
		/lib
    /lib64
    /usr/lib
    /usr/lib64
    /usr/local/lib
		/usr/local/lib64
		$ENV{NCCL_HOME}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARIES)
mark_as_advanced(NCCL_INCLUDE_DIR NCCL_LIBRARIES)