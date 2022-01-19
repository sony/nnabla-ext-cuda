// Copyright 2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

// test_async_flag.cpp

#include "gtest/gtest.h"
#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "non_stop_kernel.cuh"
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/singleton_manager.hpp>
#include <nbla/synced_array.hpp>

namespace nbla {

TEST(AsyncFalgTest, AsyncAndSafe) {
  init_cpu();
  init_cuda();

  // Make an original array
  Size_t size = 32;
  std::vector<float> orig_arr(size);
  for (int i = 0; i < size; i++) {
    orig_arr[i] = i;
  }

  // Set up an array
  auto saptr = std::make_shared<SyncedArray>(size);

  // Set up contexts
  Context host_ctx{{"cpu:float"}, "CpuCachedArray", "0"};
  Context device_ctx{{"cuda:float"}, "CudaCachedArray", "0"};
  dtypes dtype = dtypes::FLOAT;

  // Set the values to an array on host
  auto h_arr = saptr->cast_sp(dtype, host_ctx);
  auto h_arr_data = h_arr->pointer<float>();
  for (int i = 0; i < size; i++) {
    h_arr_data[i] = orig_arr[i];
  }
  ASSERT_EQ(h_arr.use_count(), 2); // h_arr and array in saptr

  // Run a kernel which stops only by passing false to flag.
  bool h_flag = true;
  bool *d_flag;
  NBLA_CUDA_CHECK(cudaMalloc(&d_flag, sizeof(bool)));
  NBLA_CUDA_CHECK(cudaMemcpy(d_flag, &h_flag, 1, cudaMemcpyHostToDevice));
  stop_null_stream_until_flag_set(d_flag);

  // Keep cudaMemcpyAsync waiting for an event of the kernel execution
  cudaEvent_t kernel_event;
  NBLA_CUDA_CHECK(cudaEventCreate(&kernel_event));
  NBLA_CUDA_CHECK(cudaEventRecord(kernel_event, 0));
  cudaStreamWaitEvent(SingletonManager::get<Cuda>()->stream_HtoD, kernel_event,
                      0);
  ASSERT_EQ(cudaEventQuery(kernel_event), cudaErrorNotReady);

  // Issue a cudaMmemcpyAsync of the values to device by using cast with async
  // and safe flags
  saptr->cast(dtype, device_ctx, false, AsyncFlag::ASYNC);
  cudaEvent_t memcpy_event;
  NBLA_CUDA_CHECK(cudaEventCreate(&memcpy_event));
  NBLA_CUDA_CHECK(cudaEventRecord(memcpy_event,
                                  SingletonManager::get<Cuda>()->stream_HtoD));
  ASSERT_EQ(cudaEventQuery(memcpy_event), cudaErrorNotReady);

  // shared_ptr were added in the event object of the cudaMemcpyAsync
  // and in the delete guard of safe flag.
  // The shared_ptr in SyncedArray was deleted by the previous saptr->cast.
  ASSERT_EQ(h_arr.use_count(), 3);

  // Make null stream wait for the cudaMmemcpyAsync. Note that host does not
  // wait for it.
  // The shared pointer of the array which is the source of the cast will be
  // deleted.
  saptr->get(dtype, device_ctx); // same as the previous cast

  // shared_ptr of the event were deleted. And that of the delete guard is still
  // alive.
  ASSERT_EQ(h_arr.use_count(), 2);

  // Stop the kernel and then the cudaMemcpyAsync will be executed.
  cudaStream_t stream;
  NBLA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  // cudaMemcpyAsync(d_flag, h_flag, 1, cudaMemcpyHostToDevice, stream);
  cudaMemsetAsync(d_flag, false, sizeof(bool), stream);

  // Wait for the end of the cudaMemcpyAsync and the removal of the delete
  // guard.
  cudaEventSynchronize(memcpy_event);
  ASSERT_EQ(h_arr.use_count(), 1);

  NBLA_CUDA_CHECK(cudaFree(d_flag));
}
}
