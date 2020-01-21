// Copyright (c) 2017 Sony Corporation. All Rights Reserved.
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

using std::vector;
using std::string;
using std::shared_ptr;

/**
Initialize CUDA features.
*/
NBLA_CUDA_API void init_cuda();

/** Clear all CUDA memory cache
*/
NBLA_CUDA_API void clear_cuda_memory_cache();

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
}
#endif
