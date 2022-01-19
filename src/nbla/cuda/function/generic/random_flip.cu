// Copyright 2019,2020,2021 Sony Corporation.
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

#include <nbla/array.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/random_flip.hpp>
#include <nbla/variable.hpp>
namespace nbla {

template <typename T>
void RandomFlipCuda<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  RandomFlip<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
  const Shape_t shape = outputs[0]->shape();
  const Shape_t stride = outputs[0]->strides();
  size_t size = outputs[0]->size();
  const int shape_info_size = shape.size() * 2;

  Context cpu_ctx{{}, "CpuCachedArray", "0"};
  this->shape_info_buf_.reshape(Shape_t{shape_info_size}, true);
  int *shape_info_cpu = this->shape_info_buf_.cast(dtypes::INT, cpu_ctx, true)
                            ->template pointer<int>();
  this->onehot_axses_.reshape(Shape_t{inputs[0]->ndim()}, true);
  int *onehot_axses_cpu = this->onehot_axses_.cast(dtypes::INT, cpu_ctx, true)
                              ->template pointer<int>();
  for (int i = 0; i < shape.size(); i++) {
    shape_info_cpu[i * 2] = shape[i];      // shape
    shape_info_cpu[i * 2 + 1] = stride[i]; // stride
    auto itr = std::find(this->axes_.begin(), this->axes_.end(), i);
    onehot_axses_cpu[i] = itr != this->axes_.end();
  }

  output_data_for_recomp_.reshape(outputs[0]->shape(), true);
}

template <typename T, bool accum>
__global__ void kernel_random_flip(const int num, const int dim, T *y,
                                   const T *x, const int *shape_info,
                                   const int *flip_flags,
                                   const int *onehot_axes, const int base_axis,
                                   const int size) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    // Determin input address
    int addr = 0, flip_index = 0;
    for (int id = 0; id < dim; id++) {
      const int shape_info_offset = id * 2;
      const int o = (idx / shape_info[shape_info_offset + 1]) // stride
                    % shape_info[shape_info_offset];          // shape
      const int i = (flip_flags[flip_index * dim + id] % 2) * onehot_axes[id]
                        ? shape_info[shape_info_offset] - 1 - o
                        : o;
      addr += i * shape_info[shape_info_offset + 1]; // stride
      if (dim < base_axis) {
        flip_index = (flip_index + 1) % size;
      }
    }
    // Copy input to output
    if (accum) {
      y[idx] += x[addr];
    } else {
      y[idx] = x[addr];
    }
  }
}

template <typename T>
void RandomFlipCuda<T>::setup_recompute_impl(const Variables &inputs,
                                             const Variables &outputs) {
  save_output_data_ = true;
}

template <typename T>
void RandomFlipCuda<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);
  this->flip_flags_.reshape(
      Shape_t{static_cast<Size_t>(this->size_ * inputs[0]->ndim())}, true);
  int *flip_flags = this->flip_flags_.cast(dtypes::INT, this->ctx_, true)
                        ->template pointer<int>();
  curandGenerator_t &gen =
      this->seed_ == -1 ? SingletonManager::get<Cuda>()->curand_generator()
                        : curand_generator_;
  curand_generate_rand<int>(gen, 0, 255, flip_flags,
                            this->size_ * inputs[0]->ndim());
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  size_t size = outputs[0]->size();
  const int *onehot_axses_gpu =
      this->onehot_axses_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();
  const int *shape_info_gpu =
      this->shape_info_buf_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_random_flip<Tcu, false>), size,
                                 inputs[0]->ndim(), y, x, shape_info_gpu,
                                 flip_flags, onehot_axses_gpu, this->base_axis_,
                                 this->size_);

  // Save output data for recomputation.
  if (save_output_data_) {
    save_output_data<Tcu>(this->ctx_, outputs[0], output_data_for_recomp_);
  }
}

template <typename T>
void RandomFlipCuda<T>::recompute_impl(const Variables &inputs,
                                       const Variables &outputs) {
  // Restore output data of previous forward execution.
  restore_output_data<Tcu>(this->ctx_, output_data_for_recomp_, outputs[0]);
  save_output_data_ = false;
}

template <typename T>
void RandomFlipCuda<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);
  int *flip_flags = this->flip_flags_.cast(dtypes::INT, this->ctx_, true)
                        ->template pointer<int>();
  Tcu *dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  const Tcu *dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  size_t size = outputs[0]->size();
  const int *onehot_axses_gpu =
      this->onehot_axses_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();
  const int *shape_info_gpu =
      this->shape_info_buf_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_random_flip<Tcu, true>), size,
                                   inputs[0]->ndim(), dx, dy, shape_info_gpu,
                                   flip_flags, onehot_axses_gpu,
                                   this->base_axis_, this->size_);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_random_flip<Tcu, false>), size,
                                   inputs[0]->ndim(), dx, dy, shape_info_gpu,
                                   flip_flags, onehot_axses_gpu,
                                   this->base_axis_, this->size_);
  }
}
}
