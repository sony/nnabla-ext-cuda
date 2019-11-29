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

#ifndef __NBLA_CUDA_EVENT_HPP__
#define __NBLA_CUDA_EVENT_HPP__

#include <nbla/event.hpp>
#include <nbla/array.hpp>
#include <nbla/cpu.hpp>

#include <cuda_runtime.h>

namespace nbla {

class CudaEvent : public Event {
  cudaEvent_t raw_event_; // Event
  ArrayPtr src_; // Source of memory copy

public:
  CudaEvent(cudaEvent_t event, ArrayPtr &src);
  CudaEvent(cudaEvent_t event, ArrayPtr &&src);
  virtual ~CudaEvent();
  virtual cudaEvent_t raw_event();

  // Return the flag which is true if this event can be delete.
  virtual bool wait_event(const Context ctx,
                          const int async_flags = AsyncFlag::NONE) override;

private:
  // Checker of CPU array class 
  inline bool is_cpu_context(const Context ctx) {
    auto cpu_array_classes = SingletonManager::get<Cpu>()->array_classes();
    
    return std::find(cpu_array_classes.begin(),
                     cpu_array_classes.end(),
                     ctx.array_class) != cpu_array_classes.end();
  }
};

typedef shared_ptr<CudaEvent> CudaEventPtr;
}
#endif