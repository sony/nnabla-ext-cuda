// Copyright 2021 Sony Corporation.
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

#include "non_stop_kernel.cuh"

namespace nbla {

__device__ int val;

__global__ void non_stop_kernel(volatile bool *flag) {
  while (flag[0]) {
    // Continue until flag turns to false.

    // Do something
    val++;
    val %= 100;
  }
}

void stop_null_stream_until_flag_set(bool *d_flag) {
  non_stop_kernel<<<1, 1>>>(d_flag);
}
} // namespace nbla
