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

#ifndef __NBLA_CUDA_REDUCE_HPP__
#define __NBLA_CUDA_REDUCE_HPP__

namespace nbla {

#define NBLA_CUDA_REDUCE_NUM_THREADS 256

/** This class can be used in Function::setup to prepare for the reduction
    CUDA kernel called in Function::forward or Function::backward.

    The input shape are integrated to the two dimension (size_y, size_x)
    according to reduction axes; one part is the collection of reduction axes
    and the other part is the collection of the other axes. However
    the terminology x and y are determined for easier understanding of
    the implemented algorithm as follow.
      - x: the integrated dimensions including memory continuous dimension.
      - y: otherwise

    For example, let an input shape (2, 3, 4, 5) and an reduction axes (0, 2).
    The dimensional part of x is (3, 5). That of y is (2, 4). Then
      - (ndim_y, ndim_x) = (2, 2)
      - (size_y, size_x) = (8, 15).

    The original strides are (60, 20, 5, 1). Then
      - strides_x_input = (20, 1)
      - strides_y_input = (60, 5)
      - strides_x = (5, 1), which is the strides of the x-part shape (3, 5)
      - strides_y = (4, 1), which is the strides of the y-part shape (2, 4)
 */
class ReduceSetup {
  // Terminology
public:
  // The constant parameters for CUDA kernel
  // See also the above comment for detail
  Size_t size_input;
  Size_t size_x, size_y;
  Size_t ndim_x, ndim_y;
  // The original strides of the input
  Shape_t strides_x_input;
  Shape_t strides_y_input;
  // The independent strides of each integrated dimension
  Shape_t strides_x;
  Shape_t strides_y;

  // The parameters for CUDA kernel launch
  bool copy_only = false;
  bool reduce_x = false;
  bool require_64bit_index = true;
  int min_blocks;

  /** Setup operator

      - Empty reduce_axes is acceptable. It makes the just copy of the input
        without reduction.
      - Negative values in reduce_axes are acceptable. The negative axis counts
        from the last to the first axis.
   */
  void operator()(const Shape_t &shape_input, const Shape_t &reduce_axes);
};
}
#endif
