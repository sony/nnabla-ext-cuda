// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef __NBLA_CUDA_INIT_HPP__
#define __NBLA_CUDA_INIT_HPP__

#include <nbla/cuda/defs.hpp>

#include <memory>
#include <string>
#include <vector>

namespace nbla {

using std::shared_ptr;
using std::string;
using std::vector;

/**
Initialize CUDA features.
*/
NBLA_CUDA_API void init_cuda();

/** Clear all CUDA memory from cache.
 */
NBLA_CUDA_API void clear_cuda_memory_cache();
/**
 * Print cache map for CUDA cached memory .
 */
NBLA_CUDA_API void print_cuda_memory_cache_map();

/**
 * APIs to analyse cache map in CUDA CachingAllocator.
 */
NBLA_CUDA_API size_t
get_cuda_caching_allocator_fragmentation_bytes(const string &device_id);
NBLA_CUDA_API size_t
get_cuda_caching_allocator_max_available_bytes(const string &device_id);
NBLA_CUDA_API vector<int>
get_cuda_cached_memory_used_counts(const string &device_id);

/**
 * Print cache map for CUDA virtual memory.
 */
NBLA_CUDA_API void print_cuda_virtual_memory_cache_map();

/**
 * Clear all CUDA virtual memory from cache.
 */
NBLA_CUDA_API void clear_cuda_virtual_memory_cache();

/**
 * APIs to analyse cache map in CUDA VirtualCachingAllocator.
 */
NBLA_CUDA_API size_t
get_cuda_virtual_caching_allocator_fragmentation_bytes(const string &device_id);
NBLA_CUDA_API size_t
get_cuda_virtual_caching_allocator_max_available_bytes(const string &device_id);
NBLA_CUDA_API vector<int>
get_cuda_virtual_memory_used_counts(const string &device_id);

/**
 * Check if tf32 is enabled or not.
 */
NBLA_CUDA_API bool is_cuda_tf32_enabled();

/** Get CUDA array classes.
 */
NBLA_CUDA_API vector<string> cuda_array_classes();

/** Set CUDA array classes
 */
NBLA_CUDA_API void _cuda_set_array_classes(const vector<string> &a);

/** Wrapper of cudaDeviceSynchronize
 */
NBLA_CUDA_API void cuda_device_synchronize(const string &device);

/** Wrapper of cudaGetDeviceCount
 */
NBLA_CUDA_API int cuda_get_device_count();

/** get available devices.
 */
NBLA_CUDA_API vector<string> cuda_get_devices();

/** cudaStream wrapper functions.
 */
NBLA_CUDA_API shared_ptr<void> cuda_create_stream(int device_id = -1);

NBLA_CUDA_API void *cuda_stream_shared_to_void(shared_ptr<void> stream);
NBLA_CUDA_API void print_stream_flag(shared_ptr<void> stream);
NBLA_CUDA_API void print_stream_priority(shared_ptr<void> stream);
NBLA_CUDA_API void cuda_stream_synchronize(shared_ptr<void> stream);
NBLA_CUDA_API void cuda_nullstream_synchronize();
NBLA_CUDA_API void cuda_stream_destroy(shared_ptr<void> stream);

/** cudaEvent wrapper functions.
 */
NBLA_CUDA_API shared_ptr<void> cuda_create_event(int device_id = -1,
                                                 unsigned int flags = 0x02);
NBLA_CUDA_API void cuda_default_stream_event(shared_ptr<void> event);
NBLA_CUDA_API void cuda_stream_wait_event(shared_ptr<void> stream,
                                          shared_ptr<void> event);
NBLA_CUDA_API void cuda_event_synchronize(shared_ptr<void> event);
NBLA_CUDA_API void cuda_event_record(shared_ptr<void> event);
NBLA_CUDA_API float cuda_event_elapsed_time(shared_ptr<void> event_s,
                                            shared_ptr<void> event_e);

/** Utils for Virtual memory allocator **/
NBLA_CUDA_API void set_cuda_vma_chunk_size(size_t size);
} // namespace nbla
#endif
