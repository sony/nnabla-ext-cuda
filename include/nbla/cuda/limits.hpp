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

#ifndef NBLA_CUDA_NUMERIC_LIMITS_HPP_
#define NBLA_CUDA_NUMERIC_LIMITS_HPP_

#include <nppdefs.h>

#include <nbla/cuda/half.hpp>

namespace nbla {
template <class T> class numeric_limits_cuda;

template <> class numeric_limits_cuda<HalfCuda> {
public:
  __device__ static HalfCuda min() { return 6.10352e-5; };
  __device__ static HalfCuda max() { return 3.2768e+4; };
};
template <> class numeric_limits_cuda<float> {
public:
  __device__ static float min() { return NPP_MINABS_32F; };
  __device__ static float max() { return NPP_MAXABS_32F; };
  //__device__ float epsilon();
  //__device__ float round_error();
  //__device__ float denorm_min();
  //__device__ float infinity();
  //__device__ float quiet_NaN();
  //__device__ float signaling_NaN();
};
}
#endif /* NBLA_CUDA_NUMERIC_LIMITS_HPP_ */
