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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/event.hpp>

namespace nbla {

CudaEvent::CudaEvent(CudaEventFlag flag) : raw_event_() {
  cudaEventCreateWithFlags(&raw_event_, flag);
}

CudaEvent::CudaEvent(cudaEvent_t event, ArrayPtr &src)
    : raw_event_(event), src_(src) {}

CudaEvent::CudaEvent(cudaEvent_t event, ArrayPtr &&src)
    : raw_event_(event), src_(src) {}

CudaEvent::~CudaEvent() { cudaEventDestroy(raw_event_); }

cudaEvent_t CudaEvent::raw_event() { return raw_event_; }

void CudaEvent::wait_event(const Context ctx, const int async_flags) {
  // Null stream (function stream) waits for an event
  NBLA_CUDA_CHECK(cudaStreamWaitEvent(0, raw_event_, 0));

  // If this event is waited for on CPU, in addition to null stream,
  // the host also wait for this event.
  if (!(async_flags & AsyncFlag::ASYNC) && !(async_flags & AsyncFlag::UNSAFE) &&
      is_cpu_context(ctx)) {
    NBLA_CUDA_CHECK(cudaStreamSynchronize(0));
  }
}

void CudaEvent::record(cudaStream_t stream) {
  NBLA_CUDA_CHECK(cudaEventRecord(raw_event_, stream));
}

void CudaEvent::sync() { NBLA_CUDA_CHECK(cudaEventSynchronize(raw_event_)); }

cudaError_t CudaEvent::query() { return cudaEventQuery(raw_event_); }
}
