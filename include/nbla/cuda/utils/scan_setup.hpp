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

#ifndef __NBLA_CUDA_UTILS_SCAN_SETUP_HPP__
#define __NBLA_CUDA_UTILS_SCAN_SETUP_HPP__

#include <nbla/common.hpp>

namespace nbla {

class ScanSetup {
public:
  Size_t size_input;
  Size_t size_outer, size_scan, size_inner;
  int axis;
  bool exclusive, reverse;
  bool accum;

  bool require_64bit_index = true;

  void operator()(const Shape_t &shape_input, const int axis,
                  const bool exclusive, const bool reverse, const bool accum);
};
}

#endif