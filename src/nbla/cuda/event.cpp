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

#include <nbla/cuda/event.hpp>
#include <nbla/cuda/common.hpp>

namespace nbla {

CudaEvent::CudaEvent(cudaEvent_t event, ArrayPtr &src)
  : raw_event_(event), src_(src) {}

CudaEvent::CudaEvent(cudaEvent_t event, ArrayPtr &&src)
  : raw_event_(event), src_(src) {}

CudaEvent::~CudaEvent() {
  cudaEventDestroy(raw_event_);
}

cudaEvent_t CudaEvent::raw_event() { return raw_event_; }

bool CudaEvent::wait_event(const Context ctx,
                           const int async_flags) {
  if (src_) {
    // Null stream (function stream) waits for an event
    NBLA_CUDA_CHECK(cudaStreamWaitEvent(0, raw_event_, 0));

    if (!(async_flags & AsyncFlag::ASYNC) &&
        (async_flags & AsyncFlag::UNSAFE) && 
        is_cpu_context(ctx)) {
      // This event will be needed for additional wait to resolve the dependency
      // on subsequent get/cast on host. This event cannot be deleted yet.
      src_ = nullptr; // Only release the memory of the source array
      return false;
    }
  }
  else {
    // Resolve the rest dependency. Host waits for this event.
    if (!(async_flags & AsyncFlag::ASYNC) &&
        !(async_flags & AsyncFlag::UNSAFE) &&
        is_cpu_context(ctx)) {
      NBLA_CUDA_CHECK(cudaStreamSynchronize(0));
    }
  }

  // This event has been finished. It can be deleted.
  return true;
}
}
