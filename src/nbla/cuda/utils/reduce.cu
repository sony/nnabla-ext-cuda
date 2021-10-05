// Copyright (c) 2021 Sony Corporation. All Rights Reserved.
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
#include <nbla/cuda/utils/reduce.hpp>

namespace nbla {

using std::vector;

/** This class perform the dimensional compression of the input shape
    to simplify the reduction computation and achieve the integrated dimension
    for the implemented reduction algorithm.

    - The adjacent dimensions with the same property (reduce or not reduce)
      can be integrated into a single dimension.
    - The dimension of size 1 can be ignored.
 */
class Compressor {
  Shape_t compressed_reduce_axes_;
  Shape_t compressed_shape_input_;

  void compress(const Shape_t &shape_input, const Shape_t &reduce_axes) {
    // Flag in the dimensions to be reduced.
    vector<bool> reduce_flags(shape_input.size(), false);
    for (const auto reduce_axis : reduce_axes) {
      reduce_flags[reduce_axis] = true;
    }

    // Compression
    Size_t compressing_size = 1;

    for (Size_t i = 0; i < reduce_flags.size() - 1; ++i) {
      compressing_size *= shape_input[i];

      if (reduce_flags[i] != reduce_flags[i + 1]) {
        // Stop compression and store the integreted dimension.
        compressed_shape_input_.push_back(compressing_size);
        if (reduce_flags[i]) {
          compressed_reduce_axes_.push_back(compressed_shape_input_.size() - 1);
        }
        compressing_size = 1; // Reset
      }
    }

    // Tail process
    if (reduce_flags.size() > 0) {
      // Stop the last compression and store the integreted dimension.
      compressing_size *= shape_input[reduce_flags.size() - 1];
      compressed_shape_input_.push_back(compressing_size);
      if (reduce_flags.back()) {
        compressed_reduce_axes_.push_back(compressed_shape_input_.size() - 1);
      }
    }
  }

  void ignore_1(Shape_t &ignored_shape_input, Shape_t &ignored_reduce_axes) {
    vector<bool> reduce_flags(compressed_shape_input_.size(), false);
    for (const auto reduce_axis : compressed_reduce_axes_) {
      reduce_flags[reduce_axis] = true;
    }

    for (Size_t i = 0; i < reduce_flags.size(); ++i) {
      if (compressed_shape_input_[i] != 1) {
        ignored_shape_input.push_back(compressed_shape_input_[i]);
        if (reduce_flags[i]) {
          ignored_reduce_axes.push_back(ignored_shape_input.size() - 1);
        }
      }
    }
  }

public:
  Compressor(const Shape_t &shape_input, const Shape_t &reduce_axes) {
    compress(shape_input, reduce_axes);
    Shape_t tmp_compressed_reduce_axes_;
    Shape_t tmp_compressed_shape_input_;
    ignore_1(tmp_compressed_shape_input_, tmp_compressed_reduce_axes_);
    compressed_reduce_axes_ = {};
    compressed_shape_input_ = {};
    compress(tmp_compressed_shape_input_, tmp_compressed_reduce_axes_);
  }

  Shape_t get_compressed_reduce_axes() const { return compressed_reduce_axes_; }
  Shape_t get_compressed_shape_input() const { return compressed_shape_input_; }
};

void ReduceSetup::operator()(const Shape_t &shape_input,
                             const Shape_t &reduce_axes) {
  size_input = std::accumulate(shape_input.cbegin(), shape_input.cend(), 1,
                               std::multiplies<Size_t>());

  // Use 32-bit indexing if possible to reduce the register consumption of
  // reduction CUDA kernel and to achieve more parallelism.
  if (size_input > std::numeric_limits<uint32_t>::max()) {
    require_64bit_index = true;
  } else {
    require_64bit_index = false;
  }

  // Debug code
  // The code path using Size_t for indexing is difficult for testing because
  // it requires the large GPU memory usage. The following code can be used
  // to test the Size_t code path for developer.
  // std::cout << "Force Size_t" << std::endl;
  // require_64bit_index_ = true;

  // If reduction is not required, just copy the input.
  if (reduce_axes.size() == 0 || size_input == 1) {
    copy_only = true;
    return;
  }

  // Convert the negative axes to the positives.
  const Size_t ndim = static_cast<Size_t>(shape_input.size());
  Shape_t positive_reduce_axes;
  for (Size_t a : reduce_axes) {
    if (a < 0) {
      a += ndim;
    }
    NBLA_CHECK(a < ndim && a >= 0, error_code::value,
               "Axes out of range. 0 <= %d < %d", a, ndim);
    positive_reduce_axes.push_back(a);
  }

  // Integrate the reduce axes.
  const Compressor compressor(shape_input, positive_reduce_axes);
  const auto compressed_reduce_axes = compressor.get_compressed_reduce_axes();
  const auto compressed_shape_input = compressor.get_compressed_shape_input();
  const auto compressed_ndim_input = compressed_shape_input.size();
  const auto compressed_strides_input =
      get_c_contiguous_strides(compressed_shape_input);

  // Re-check if reduction is required.
  if (compressed_reduce_axes.size() == 0) {
    copy_only = true;
    return;
  }

  // Determine the shape parameters for the implemented algorithm.
  reduce_x = (compressed_reduce_axes.back() == compressed_ndim_input - 1);
  size_x = size_y = 1;
  ndim_x = ndim_y = 0;
  Shape_t shape_x, shape_y;

  for (Size_t i = compressed_ndim_input - 1; i >= 0; i -= 2) {
    size_x *= compressed_shape_input[i];
    ndim_x++;
    shape_x.insert(shape_x.begin(), compressed_shape_input[i]);
  }
  for (Size_t i = compressed_ndim_input - 2; i >= 0; i -= 2) {
    size_y *= compressed_shape_input[i];
    ndim_y++;
    shape_y.insert(shape_y.begin(), compressed_shape_input[i]);
  }

  strides_x = get_c_contiguous_strides(shape_x);
  strides_y = get_c_contiguous_strides(shape_y);
  strides_x_input = Shape_t(ndim_x);
  strides_y_input = Shape_t(ndim_y);
  int x_idx = 1;
  int y_idx = 1;
  for (Size_t i = compressed_ndim_input - 1; i >= 0; i -= 2) {
    strides_x_input[ndim_x - x_idx++] = compressed_strides_input[i];
  }
  for (Size_t i = compressed_ndim_input - 2; i >= 0; i -= 2) {
    strides_y_input[ndim_y - y_idx++] = compressed_strides_input[i];
  }

  // Collect the information about the device
  const auto device_prop = cuda_get_current_device_properties();
  const auto min_blocks_per_sm =
      device_prop.maxThreadsPerMultiProcessor / NBLA_CUDA_REDUCE_NUM_THREADS;
  min_blocks = min_blocks_per_sm * device_prop.multiProcessorCount;
}
}
