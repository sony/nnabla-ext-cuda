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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/random_crop.hpp>
#include <nbla/variable.hpp>

namespace nbla {

template <typename T>
void RandomCropCuda<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  RandomCrop<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  const Shape_t shape = outputs[0]->shape();
  const Shape_t stride = outputs[0]->strides();
  size_t size = outputs[0]->size();
  const int shape_info_size = shape.size() * 5;

  Context cpu_ctx{{}, "CpuCachedArray", "0"};
  this->shape_info_buf_.reshape(Shape_t{shape_info_size}, true);

  int *shape_info_cpu = this->shape_info_buf_.cast(dtypes::INT, cpu_ctx, true)
                            ->template pointer<int>();
  for (int i = 0; i < shape.size(); i++) {
    shape_info_cpu[i * 5] = shape[i];      // output_shape
    shape_info_cpu[i * 5 + 1] = stride[i]; // output_stride
    shape_info_cpu[i * 5 + 2] =
        i >= this->dim_offset_
            ? inputs[0]->shape()[i] - this->shape_[i - this->dim_offset_] + 1
            : 0;                                         // crop indexs
    shape_info_cpu[i * 5 + 3] = inputs[0]->shape()[i];   // input_shape
    shape_info_cpu[i * 5 + 4] = inputs[0]->strides()[i]; // input_stride
  }

  output_data_for_recomp_.reshape(outputs[0]->shape(), true);
}

template <typename T, bool is_backward>
__global__ void kernel_random_crop(const int num, const int dim, T *dst,
                                   const T *src, const int *shape_info,
                                   const int *random_values,
                                   const int base_axis, const int size,
                                   const int crop_size, const int dim_offset) {

#define OUT_SHAPE(dim) shape_info[dim * 5]
#define OUT_STRIDE(dim) shape_info[dim * 5 + 1]
#define LEFT(dim) shape_info[dim * 5 + 2]
#define IN_SHAPE(dim) shape_info[dim * 5 + 3]
#define IN_STRIDE(dim) shape_info[dim * 5 + 4]

  NBLA_CUDA_KERNEL_LOOP(idx, num) {

    int jdx = 0, random_index = 0, offset = 0;

    // Determin input offset
    for (int id = 0; id < dim; id++) {
      const int o = (idx / OUT_STRIDE(id)) % OUT_SHAPE(id);
      offset += o * (IN_STRIDE(id) - OUT_STRIDE(id));
    }

    // Determin input address
    for (int id = 0; id < dim; id++) {
      const int o = ((idx + offset) / IN_STRIDE(id)) % IN_SHAPE(id);
      const int left =
          id >= dim_offset
              ? random_values[random_index * crop_size + id - dim_offset] %
                    LEFT(id)
              : 0;

      jdx += (o + left) * IN_STRIDE(id);

      if (id < base_axis) {
        random_index = (random_index + 1) % size;
      }
    }

    // Note src is x and dst is y in forward,
    // while src is dy and dst is dx in backward.

    if (is_backward) {
      // In backward, dy is accumulated to dx.
      dst[idx] += src[jdx];
    } else {
      // In for forward, x is copied to y.
      dst[idx] = src[jdx];
    }
  }

#undef OUT_SHAPE
#undef OUT_STRIDE
#undef LEFT
#undef IN_SHAPE
#undef IN_STRIDE
}

template <typename T>
void RandomCropCuda<T>::setup_recompute_impl(const Variables &inputs,
                                             const Variables &outputs) {
  save_output_data_ = true;
}

template <typename T>
void RandomCropCuda<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);
  size_t size = outputs[0]->size();
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  this->random_values_ = make_shared<CudaCachedArray>(
      this->size_ * this->shape_.size(), dtypes::INT, this->ctx_);
  int *random_values = this->random_values_->template pointer<int>();
  curandGenerator_t &gen =
      this->seed_ == -1 ? SingletonManager::get<Cuda>()->curand_generator()
                        : curand_generator_;
  curand_generate_rand<int>(gen, 0, 1 << 23, random_values,
                            this->size_ * this->shape_.size());
  const int *shape_info_gpu =
      this->shape_info_buf_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_random_crop<Tcu, false>), size,
                                 inputs[0]->ndim(), y, x, shape_info_gpu,
                                 random_values, this->base_axis_, this->size_,
                                 this->shape_.size(), this->dim_offset_);

  // Save output data for recomputation.
  if (save_output_data_) {
    save_output_data<Tcu>(this->ctx_, outputs[0], output_data_for_recomp_);
  }
}

template <typename T>
void RandomCropCuda<T>::recompute_impl(const Variables &inputs,
                                       const Variables &outputs) {
  // Restore output data of previous forward execution.
  restore_output_data<Tcu>(this->ctx_, output_data_for_recomp_, outputs[0]);
  save_output_data_ = false;
}

template <typename T>
void RandomCropCuda<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);

  size_t size = outputs[0]->size();
  if (!accum[0])
    inputs[0]->grad()->zero();
  Tcu *dx = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
  const Tcu *dy = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);

  int *random_values = this->random_values_->template pointer<int>();
  const int *shape_info_gpu =
      this->shape_info_buf_.get(dtypes::INT, this->ctx_)
          ->template const_pointer<int>();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_random_crop<Tcu, true>), size,
                                 inputs[0]->ndim(), dx, dy, shape_info_gpu,
                                 random_values, this->base_axis_, this->size_,
                                 this->shape_.size(), this->dim_offset_);
}
}
