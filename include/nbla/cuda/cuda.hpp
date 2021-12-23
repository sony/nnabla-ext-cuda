// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
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

/** Cpu resources
*/
#ifndef __NBLA_CUDA_CUDA_HPP__
#define __NBLA_CUDA_CUDA_HPP__

#include <nbla/backend_base.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/defs.hpp>
#include <nbla/cuda/init.hpp>
#include <nbla/cuda/memory/cuda_memory.hpp>
#include <nbla/exception.hpp>
#include <nbla/memory/allocator.hpp>
#include <nbla/singleton_manager.hpp>

// Todo: avoid including cudnn.h in cuda package.
#include <cudnn.h>

#include <mutex>
#include <thread>
#include <unordered_map>

namespace nbla {

using std::unordered_map;

/**
 * Enum for nbla global streams.
 */
enum CudaStreamId { CONVOLUTION_BWD, MAX_COUNT };

/**
Singleton class for storing some handles or configs for CUDA Computation.
*/
class NBLA_CUDA_API Cuda : public BackendBase {

public:
  ~Cuda();
  /** Get cuBLAS handle of a specified device */
  cublasHandle_t cublas_handle(int device = -1);

  /** Get or create cuda event */
  std::shared_ptr<cudaEvent_t> cuda_event(unsigned int flags, int device = -1);

  /** Get cuRAND global generator **/
  curandGenerator_t &curand_generator();

  void curand_set_seed(int seed);

  template <typename T> void curand_generate_uniform(T *r, int size);

  /** Available array class list used in CUDA Function implementations.
   */
  vector<string> array_classes() const;

  /** Set array class list.

      @note Dangerous to call. End users shouldn't call.
   */
  void _set_array_classes(const vector<string> &a);

  /** Register array class to available list by name.
   */
  void register_array_class(const string &name);

  /** Get a caching allocator.
   */
  shared_ptr<Allocator> caching_allocator();

  /** Get a no-cache allocator.
   */
  shared_ptr<Allocator> naive_allocator();

  /** Get a caching unified-memory allocator.
   */
  shared_ptr<Allocator> unified_allocator();

  /** Get a caching pinned-host-memory allocator.
   */
  shared_ptr<Allocator> pinned_allocator();

  /** Get a caching virtual-memory allocator.
 */
  shared_ptr<Allocator> virtual_caching_allocator();

  /** Free all unused host memory caches
  */
  void free_unused_host_caches();

  /** Synchronize host to device.
   */
  void device_synchronize(const string &device);

  /** Synchronize host to default stream of device.
   */
  void default_stream_synchronize(const string &device);

  /** Get auxilliary stream
   */
  shared_ptr<cudaStream_t> get_stream(unsigned int flag, CudaStreamId streamId,
                                      int device = -1);

  /** Non blockuing streams for data transfer
   */
  cudaStream_t stream_HtoD = 0;
  cudaStream_t stream_DtoH = 0;

  /** Create non blockuing streams for data transfer
   */
  void create_lms_streams(int device = -1);

  /** Change a chunk size of physical memory used in virtual memory allocator.
   */
  void set_vma_chunk_size(size_t size);

  /** get cuda device property
   */
  shared_ptr<cudaDeviceProp> get_device_properties(int device = -1);

protected:
  std::mutex mtx_cublas_;
  std::mutex mtx_curand_;
  std::mutex mtx_stream_;
  typedef unordered_map<std::thread::id, cublasHandle_t> tid_cublas_handle_t;
  unordered_map<int, tid_cublas_handle_t>
      cublas_handles_; ///< cuBLAS handles for each device.
  unordered_map<int, curandGenerator_t> curand_generators_;
  unordered_map<int, unordered_map<unsigned int, vector<cudaEvent_t>>>
      cuda_unused_events_;
  vector<string> array_classes_;        ///< Available array classes
  unordered_map<int, int> seed_counts_; /// this is used to check seed update

  /*
    NOTE: Allocators must be retained as shared_ptr in order to be passed to a
    CachedMemory instance to prevent destroying allocators before destroying
    memory.
   */
  shared_ptr<Allocator> naive_allocator_;
  shared_ptr<Allocator> caching_allocator_;
  shared_ptr<Allocator> unified_allocator_;
  shared_ptr<Allocator> pinned_allocator_;
#if CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000
  shared_ptr<Allocator> virtual_caching_allocator_;
#endif // CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000

  // stream pool -> <device, <id, <t_id, stream>>>
  typedef unordered_map<std::thread::id, shared_ptr<cudaStream_t>>
      tid_cuda_stream_t;
  unordered_map<int, unordered_map<int, tid_cuda_stream_t>> streams_;

  shared_ptr<cudaDeviceProp> cuda_device_prop_;

private:
  friend SingletonManager;
  // Never called by users.
  Cuda();
  DISABLE_COPY_AND_ASSIGN(Cuda);
};
}
#endif
