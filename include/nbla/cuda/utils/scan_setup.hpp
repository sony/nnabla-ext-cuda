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

/**
 * @brief Scan configulation class.
 *
 * size_input: Total size of the input.
 * size_outer: Total product of input_shape[0:axis-1]
 * size_scan: input_shape[axis]
 * size_innter: Total product of inpute_shape[axis+1:]
 * e.g.
 *  For the input_shape == (2, 3, 5, 7) and axis == 2
 *  size_input: 210
 *  size_outer: 6
 *  size_scan: 5
 *  size_inner: 7
 *
 * axis: Axis scan will be performed.
 *
 * exclusive: output[i] includes input[i] or not.
 * e.g.) exclusive == false: [1, 2, 3, 4] -> [1, 3, 6, 10]
 * e.g.) exclusive == true: [1, 2, 3, 4] -> [0, 1, 3, 6]
 *
 * reverse: Scan direction.
 * e.g.) reverse == false: [1, 2, 3, 4] -> [1, 3, 6, 10]
 * e.g.) reverse == true: [1, 2, 3, 4] -> [10, 9, 7, 4]
 *
 * require_64bit_index:
 *
 */
class ScanSetup {
public:
  Size_t size_input;
  Size_t size_outer, size_scan, size_inner;
  int axis;
  bool exclusive, reverse;
  bool require_64bit_index = true;

  /**
   * @brief Initialize member variables from input_shape and other scan
   * conditions.
   */
  void operator()(const Shape_t &shape_input, const int axis,
                  const bool exclusive, const bool reverse);
};
}

#endif