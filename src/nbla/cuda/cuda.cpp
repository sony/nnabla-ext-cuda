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

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/utils/random.hpp>
#include <nbla/singleton_manager-internal.hpp>

#include <nbla/cuda/memory/cuda_memory.hpp>
#include <nbla/cuda/memory/cuda_virtual_memory.hpp>

#include <nbla/memory/caching_allocator_with_buckets.hpp>
#include <nbla/memory/naive_allocator.hpp>
#include <nbla/memory/virtual_caching_allocator.hpp>

namespace nbla {

Cuda::Cuda()
    : naive_allocator_(make_shared<NaiveAllocator<CudaMemory>>()),
      caching_allocator_(
          make_shared<CachingAllocatorWithBuckets<CudaMemory>>()),
      unified_allocator_(
          make_shared<CachingAllocatorWithBuckets<CudaUnifiedMemory>>()),
      pinned_allocator_(
          make_shared<CachingAllocatorWithBuckets<CudaPinnedHostMemory>>())
#if CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000
      ,
      virtual_caching_allocator_(
          make_shared<
              VirtualCachingAllocator<CudaPhysicalMemory, CudaVirtualMemory>>())
#endif // CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000
{
}

Cuda::~Cuda() {
  for (auto handle : this->cublas_handles_) {
    NBLA_CUBLAS_CHECK(cublasDestroy(handle.second));
  }
  for (auto gen : this->curand_generators_) {
    curand_destroy_generator(gen.second);
  }
  for (auto &all_events : this->cuda_unused_events_) {
    for (auto &events : all_events.second) {
      for (auto &event : events.second) {
        NBLA_CUDA_CHECK(cudaEventDestroy(event));
      }
    }
  }

  for (auto &all_streams : this->streams_) {
    for (auto &tid_stream : all_streams.second) {
      for (auto &stream : tid_stream.second) {
        NBLA_CUDA_CHECK(cudaStreamDestroy(*(stream.second)));
      }
    }
  }

  if (stream_HtoD != 0) {
    NBLA_CUDA_CHECK(cudaStreamDestroy(stream_HtoD));
  }

  if (stream_DtoH != 0) {
    NBLA_CUDA_CHECK(cudaStreamDestroy(stream_DtoH));
  }
}

cublasHandle_t Cuda::cublas_handle(int device) {
  if (device < 0) {
    device = cuda_get_device();
  }
  std::lock_guard<decltype(mtx_cublas_)> lock(mtx_cublas_);
  auto it = this->cublas_handles_.find(device);
  // Create a new one
  if (it == this->cublas_handles_.end()) {
    cublasHandle_t handle;
    NBLA_CUBLAS_CHECK(cublasCreate(&handle));
    this->cublas_handles_.insert({device, handle});
    return handle;
  }
  return it->second;
}

std::shared_ptr<cudaEvent_t> Cuda::cuda_event(unsigned int flags, int device) {
  if (device < 0) {
    device = cuda_get_device();
  }

  /* Find an unused cudaEvent_t with the proper flags and device. */
  std::lock_guard<decltype(this->mtx_event_)> lock(this->mtx_event_);
  auto all_events = this->cuda_unused_events_.find(device);
  if (all_events == this->cuda_unused_events_.end()) {
    /* Insert an empty set if there is no set corresponding to the device. */
    this->cuda_unused_events_.insert(
        {device, unordered_map<unsigned int, vector<cudaEvent_t>>()});
    all_events = this->cuda_unused_events_.find(device);
  }

  auto it = all_events->second.find(flags);
  if (it == all_events->second.end()) {
    /* Insert an empty set if there is no set corresponding to the flags. */
    all_events->second.insert({flags, {}});
    it = all_events->second.find(flags);
  }

  cudaEvent_t event;
  if (it->second.empty()) {
    /* Create a new cudaEvent_t */
    NBLA_CUDA_CHECK(cudaEventCreateWithFlags(&event, flags));
  } else {
    /* Re-use the existing unused cudaEvent_t */
    event = it->second.back();
    it->second.pop_back();
  }

  std::default_delete<cudaEvent_t> deleter;
  return std::shared_ptr<cudaEvent_t>(
      new cudaEvent_t(event), [this, device, flags, deleter](cudaEvent_t *ptr) {
        /* This lambda function is a custom deleter of the std::shared_ptr.
         * It is invoked when deleting the managed cudaEvent_t.
         */

        /* Prepare empty set if there is no corresponding set */
        std::lock_guard<decltype(this->mtx_event_)> lock(this->mtx_event_);
        auto all_events = this->cuda_unused_events_.find(device);
        if (all_events == this->cuda_unused_events_.end()) {
          this->cuda_unused_events_.insert(
              {device, unordered_map<unsigned int, vector<cudaEvent_t>>()});
          all_events = this->cuda_unused_events_.find(device);
        }

        auto it = all_events->second.find(flags);
        if (it == all_events->second.end()) {
          all_events->second.insert({flags, {}});
          it = all_events->second.find(flags);
        }

        /* Push the managed cudaEvent_t to the unused set because nobody uses
         * this cudaEvent_t. */
        it->second.push_back(*ptr);

        /* Delete the raw pointer of the cudaEvent_t. */
        deleter(ptr);
      });
}

void Cuda::create_lms_streams(int device) {
  if (device < 0) {
    device = cuda_get_device();
  }

  cuda_set_device(device);

  NBLA_CUDA_CHECK(
      cudaStreamCreateWithFlags(&stream_HtoD, cudaStreamNonBlocking));
  NBLA_CUDA_CHECK(
      cudaStreamCreateWithFlags(&stream_DtoH, cudaStreamNonBlocking));
}

#if CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000
void Cuda::set_vma_chunk_size(size_t size) {
  std::dynamic_pointer_cast<VirtualCachingAllocatorBase>(
      virtual_caching_allocator_)
      ->set_chunk_size(size);
}
#else
// dummy for cython
void Cuda::set_vma_chunk_size(size_t size) {}
#endif // CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000

shared_ptr<cudaStream_t> Cuda::get_stream(unsigned int flags,
                                          CudaStreamId streamId, int device) {
  if (device < 0) {
    device = cuda_get_device();
  }

  auto tid = std::this_thread::get_id();
  std::lock_guard<std::mutex> lock(mtx_stream);

  int streamIdInt = static_cast<int>(streamId);
  auto &device_streams = this->streams_[device];

  if (device_streams.find(streamIdInt) != device_streams.end()) {
    auto &tid_cuda_stream = device_streams[streamIdInt];

    // Stream has already been created.
    if (tid_cuda_stream.find(tid) != tid_cuda_stream.end()) {
      // check flags
      auto stream = tid_cuda_stream[tid];
      unsigned int register_flags;
      NBLA_CUDA_CHECK(cudaStreamGetFlags(*stream, &register_flags));
      NBLA_CHECK(flags == register_flags, error_code::value,
                 "flag mismatch. StreamId: %u, flags created before: %u, flags "
                 "requested: %u",
                 streamId, register_flags, flags);
      return stream;
    } else {
      auto stream = shared_ptr<cudaStream_t>(new cudaStream_t());
      NBLA_CUDA_CHECK(cudaStreamCreateWithFlags(stream.get(), flags));
      device_streams[streamIdInt].insert({tid, stream});
      return stream;
    }
  } else {
    // Create stream.
    auto stream = shared_ptr<cudaStream_t>(new cudaStream_t());
    NBLA_CUDA_CHECK(cudaStreamCreateWithFlags(stream.get(), flags));
    tid_cuda_stream_t tid_cuda_stream;
    tid_cuda_stream[tid] = stream;
    this->streams_[device].insert({streamIdInt, tid_cuda_stream});
    return stream;
  }
}

curandGenerator_t Cuda::curand_generator() {
  // Get current device
  int device = cuda_get_device();
  std::lock_guard<decltype(mtx_curand_)> lock(mtx_curand_);
  // Find device rng
  auto it = this->curand_generators_.find(device);
  // Create a new one
  if (it == this->curand_generators_.end()) {
    curandGenerator_t gen = curand_create_generator();
    this->curand_generators_.insert({device, gen});
    return gen;
  }
  return it->second;
}

void Cuda::curand_set_seed(int seed) {
  ::nbla::curand_set_seed(curand_generator(), seed);
}

template <> void Cuda::curand_generate_uniform<float>(float *r, int size) {
  ::nbla::curand_generate_rand(curand_generator(), float(0), float(1), r, size);
}

vector<string> Cuda::array_classes() const { return array_classes_; }

void Cuda::_set_array_classes(const vector<string> &a) { array_classes_ = a; }

void Cuda::register_array_class(const string &name) {
  array_classes_.push_back(name);
}

shared_ptr<Allocator> Cuda::caching_allocator() { return caching_allocator_; }
shared_ptr<Allocator> Cuda::naive_allocator() { return naive_allocator_; }
shared_ptr<Allocator> Cuda::unified_allocator() { return unified_allocator_; }
shared_ptr<Allocator> Cuda::pinned_allocator() { return pinned_allocator_; }
#if CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000
shared_ptr<Allocator> Cuda::virtual_caching_allocator() {
  return virtual_caching_allocator_;
}
#endif // CUDA_VERSION >= 10020 && CUDNN_VERSION >= 8000

void Cuda::free_unused_host_caches() {
  pinned_allocator_->free_unused_caches();
  unified_allocator_->free_unused_caches();
}

void Cuda::device_synchronize(const string &device) {
  cuda_device_synchronize(device);
}

void Cuda::default_stream_synchronize(const string &device) {
  cuda_nullstream_synchronize();
}

NBLA_INSTANTIATE_SINGLETON(NBLA_CUDA_API, Cuda);
}
