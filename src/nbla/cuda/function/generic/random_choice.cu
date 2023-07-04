// Copyright 2019,2020,2021 Sony Corporation.
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

#include <nbla/array.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/random_choice.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/cuda/utils/scan_ops/sum.cuh>
#include <nbla/variable.hpp>

#include <thrust/device_vector.h>

namespace nbla {

namespace random_choice_cuda {

// CUDA kernel to uniformly draw samples from the cumulative summed weights
// (per population) in w_sums. Each population has w_size elements. The number
// of samples to draw (per population) is given by u_size, the u_vals pointer
// is input with uniform random value [0..1). Each thread (subject to grid
// striding) determines for one input element and all samples if the weight sum
// is less than the uniform value and, if true, increses the mapping index for
// that input value. After all threads have run, the index map points to the
// values drawn from input x.
template <typename T>
__global__ void draw_samples(const size_t size, const size_t w_size,
                             const size_t u_size, const T *w_sums,
                             const float *u_vals, int *idxmap) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    const auto b = i / w_size; // population index
    const auto scale = w_sums[(b + 1) * w_size - 1];

    for (int j = 0; j < u_size; j++) {
      if (w_sums[i] < u_vals[b * u_size + j] * scale) {
        atomic_add(idxmap + b * u_size + j, 1);
      }
    }
  }
}

// Same kernel as above but draws one sample per round `r`. Needed for sampling
// without replacement.
template <typename T>
__global__ void draw_sample(const size_t size, const size_t w_size,
                            const size_t u_size, const T *w_sums,
                            const float *u_vals, int *idxmap, const int r) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    const auto b = i / w_size; // population index
    const auto scale = w_sums[(b + 1) * w_size - 1];

    if (w_sums[i] < u_vals[b * u_size + r] * scale) {
      atomic_add(idxmap + b * u_size + r, 1);
    }
  }
}

// Kernel that sets choosen weigths to zero after each round `r`. Used for
// sampling without replacement.
template <typename T>
__global__ void zero_weight(const size_t size, const size_t w_size,
                            const size_t u_size, const int *idxmap, const int r,
                            T *w_data) {
  NBLA_CUDA_KERNEL_LOOP(b, size) {
    w_data[b * w_size + idxmap[b * u_size + r]] = 0;
  }
}

// Copy choosen sample values from input `x`, using the index map.
template <typename T>
__global__ void copy_samples(const size_t size, const size_t w_size,
                             const size_t u_size, const int *idxmap,
                             const T *src, T *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    dst[i] = src[(i / u_size) * w_size + idxmap[i]];
  }
}

// Backward kernel just adds gradient through index map determined at forward.
template <typename T>
__global__ void add_gradient(const size_t size, const size_t w_size,
                             const size_t u_size, const int *idxmap,
                             const T *src, T *dst) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    atomic_add(dst + (i / u_size) * w_size + idxmap[i], src[i]);
  }
}

} // namespace random_choice_cuda

template <typename T>
void RandomChoiceCuda<T>::setup_impl(const Variables &inputs,
                                     const Variables &outputs) {
  RandomChoice<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
  auto w_size = inputs.at(1)->shape().back(); // size of weight vectors
  scan_setup_(Shape_t{this->outer_loop_, w_size}, 1, false, false);
}

template <typename T>
void RandomChoiceCuda<T>::setup_recompute_impl(const Variables &inputs,
                                               const Variables &outputs) {
  save_output_data_ = true;
  output_data_for_recomp_.reshape(outputs.at(0)->shape(), true);
}

template <typename T>
void RandomChoiceCuda<T>::forward_impl(const Variables &inputs,
                                       const Variables &outputs) {
  cuda_set_device(this->device_);
  if (this->replace_ == true) {
    this->sample_with_replacement(inputs, outputs);
  } else {
    this->sample_without_replace(inputs, outputs);
  }

  // Save output data for recomputation.
  if (save_output_data_) {
    save_output_data<Tcu>(this->ctx_, outputs.at(0), output_data_for_recomp_);
  }
}

template <typename T>
void RandomChoiceCuda<T>::recompute_impl(const Variables &inputs,
                                         const Variables &outputs) {
  // Restore output data of previous forward execution.
  restore_output_data<Tcu>(this->ctx_, output_data_for_recomp_, outputs.at(0));
  save_output_data_ = false;
}

template <typename T>
void RandomChoiceCuda<T>::sample_with_replacement(const Variables &inputs,
                                                  const Variables &outputs) {
  using f32 = typename CudaTypeForceFloat<Tcu>::type;
  auto x = inputs.at(0), w = inputs.at(1), y = outputs.at(0);
  Variable &idxbuf_ = this->idxbuf_;
  idxbuf_.data()->zero();

  auto idxbuf = idxbuf_.cast_data_and_get_pointer<int>(this->ctx_, false);
  auto x_data = x->get_data_pointer<Tcu>(this->ctx_);
  auto w_data = w->get_data_pointer<f32>(this->ctx_);
  auto y_data = y->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  auto w_size = w->shape().back(); // size of each weight vector
  auto u_size = this->inner_loop_; // samples to draw per weight vector

  NdArray tmp0(Shape_t{x->size()});
  NdArray tmp1(Shape_t{y->size()});
  auto w_sums = tmp0.cast(get_dtype<f32>(), this->ctx_, true)->pointer<f32>();
  auto u_vals = tmp1.cast(get_dtype<f32>(), this->ctx_, true)->pointer<f32>();

  // Generate random choices for each output sample point.
  curandGenerator_t &gen =
      this->seed_ == -1 ? SingletonManager::get<Cuda>()->curand_generator()
                        : curand_generator_;
  curand_generate_rand<f32>(gen, 0, 1, u_vals, y->size());

  // Build cumulative sum of population weights.
  device_cumsum(this->ctx_, w_data, w_sums, scan_setup_, false /* accum */);

  // Indirectly draw samples by building an index map.
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(random_choice_cuda::draw_samples, x->size(),
                                 w_size, u_size, w_sums, u_vals, idxbuf);

  // Copy input data values according to index map.
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(random_choice_cuda::copy_samples, y->size(),
                                 w_size, u_size, idxbuf, x_data, y_data);
}

template <typename T>
void RandomChoiceCuda<T>::sample_without_replace(const Variables &inputs,
                                                 const Variables &outputs) {
  using f32 = typename CudaTypeForceFloat<Tcu>::type;
  auto x = inputs.at(0), w = inputs.at(1), y = outputs.at(0);
  Variable &idxbuf_ = this->idxbuf_;
  idxbuf_.data()->zero();

  auto idxbuf = idxbuf_.cast_data_and_get_pointer<int>(this->ctx_, false);
  auto x_data = x->get_data_pointer<Tcu>(this->ctx_);
  auto y_data = y->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
  auto w_size = w->shape().back(); // size of each weight vector
  auto u_size = this->inner_loop_; // samples to draw per weight vector
  auto b_size = this->outer_loop_; // batch size (left N-1 dims of x and w)

  NdArray tmp0(Shape_t{x->size()});
  NdArray tmp1(Shape_t{x->size()});
  NdArray tmp2(Shape_t{y->size()});
  auto w_data = tmp0.cast(get_dtype<f32>(), this->ctx_, true)->pointer<f32>();
  auto w_sums = tmp1.cast(get_dtype<f32>(), this->ctx_, true)->pointer<f32>();
  auto u_vals = tmp2.cast(get_dtype<f32>(), this->ctx_, true)->pointer<f32>();

  // Copy the weight data to writable memory where we can remove a
  // category (by nulling it's weight) after each round.
  auto w_data_ptr = w->get_data_pointer<f32>(this->ctx_);
  thrust::copy_n(thrust::device_pointer_cast(w_data_ptr), w->size(),
                 thrust::device_pointer_cast(w_data));

  // Generate random choices for each output sample point.
  curandGenerator_t &gen =
      this->seed_ == -1 ? SingletonManager::get<Cuda>()->curand_generator()
                        : curand_generator_;
  curand_generate_rand<f32>(gen, 0, 1, u_vals, y->size());

  // We draw one sample per round (and population) and set the choosen weight
  // to zero, so each round decreases the number of non-zero weights.
  for (int r = 0; r < u_size; r++) {
    // Build cumulative sum of population weights.
    device_cumsum(this->ctx_, w_data, w_sums, scan_setup_, false /* accum */);

    // Indirectly draw samples by building an index map.
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(random_choice_cuda::draw_sample, x->size(),
                                   w_size, u_size, w_sums, u_vals, idxbuf, r);

    // Set weight to zero for the batch-size number of samples drawn.
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(random_choice_cuda::zero_weight, b_size,
                                   w_size, u_size, idxbuf, r, w_data)
  }

  // Copy input data values according to index map.
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(random_choice_cuda::copy_samples, y->size(),
                                 w_size, u_size, idxbuf, x_data, y_data);
}

template <typename T>
void RandomChoiceCuda<T>::backward_impl(const Variables &inputs,
                                        const Variables &outputs,
                                        const vector<bool> &propagate_down,
                                        const vector<bool> &accum) {
  if (!(propagate_down.at(0) || propagate_down.at(1))) {
    return;
  }

  cuda_set_device(this->device_);
  auto x = inputs.at(0), w = inputs.at(1), y = outputs.at(0);
  Variable &idxbuf_ = this->idxbuf_;

  if ((propagate_down.at(0)) && (!accum.at(0)))
    x->grad()->zero();

  if ((propagate_down.at(1)) && (!accum.at(1)))
    w->grad()->zero();

  auto w_size = w->shape().back(); // size of each weight vector
  auto u_size = this->inner_loop_; // samples to draw per weight vector

  if (propagate_down.at(0)) {
    auto x_grad = x->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    auto y_grad = y->get_grad_pointer<Tcu>(this->ctx_);
    auto idxbuf = idxbuf_.get_data_pointer<int>(this->ctx_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(random_choice_cuda::add_gradient, y->size(),
                                   w_size, u_size, idxbuf, y_grad, x_grad);
  }

  if (propagate_down.at(1)) {
    auto w_grad = w->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    auto y_grad = y->get_grad_pointer<Tcu>(this->ctx_);
    auto idxbuf = idxbuf_.get_data_pointer<int>(this->ctx_);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(random_choice_cuda::add_gradient, y->size(),
                                   w_size, u_size, idxbuf, y_grad, w_grad);
  }
}
} // namespace nbla
