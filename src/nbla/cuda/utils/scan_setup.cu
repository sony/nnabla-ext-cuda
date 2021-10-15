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

#include <nbla/cuda/utils/scan_setup.hpp>

namespace nbla {
void ScanSetup::operator()(const Shape_t &shape_input, const int axis,
                           const bool exclusive, const bool reverse,
                           const bool accum) {
  this->axis = axis;
  this->exclusive = exclusive;
  this->reverse = reverse;
  this->accum = accum;

  const auto ndim = shape_input.size();
  NBLA_CHECK(0 <= axis && axis < ndim, error_code::value,
             "Axes out of range. 0 <= %d < %d", axis, ndim);

  size_outer = 1;
  for (int i = 0; i < axis; i++) {
    size_outer *= shape_input[i];
  }

  size_scan = shape_input[axis];

  size_inner = 1;
  for (int i = axis + 1; i < ndim; i++) {
    size_inner *= shape_input[i];
  }

  size_input = size_outer * size_scan * size_inner;

  // Use 32-bit indexing if possible to reduce the register consumption of
  // reduction CUDA kernel and to achieve more parallelism.
  if (size_input > std::numeric_limits<uint32_t>::max()) {
    require_64bit_index = true;
  } else {
    require_64bit_index = false;
  }
}
}
