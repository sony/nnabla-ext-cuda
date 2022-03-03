// Copyright 2018,2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/inq_affine.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/variable.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>

namespace nbla {

template <typename T, typename T1>
__global__ void kernel_copy_fixedweights(const int size, const T *old_weights,
                                         const T1 *old_indicators, T *weights) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    if (old_indicators[i] == 1) {
      weights[i] = old_weights[i];
    }
  }
}

template <typename T, typename T1>
__global__ void kernel_random_selection(const int size, T1 *indicators,
                                        T *uniform_rand) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    if (indicators[i] == 0) {
      indicators[i] = uniform_rand[i] > 0.5;
    }
  }
}

template <typename T, typename T1>
__global__ void kernel_fix_largestabs(const int size, T1 *indicators,
                                      int *indices) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { indicators[indices[i]] = 1; }
}

template <typename T, typename T1>
__global__ void kernel_quantize_weights(const int size, T1 *indicators,
                                        T *weights, int n1, int n2,
                                        T pruning_threshold) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    if (indicators[i] == 1) {
      if (abs(weights[i]) < pruning_threshold) {
        weights[i] = 0.0f;
      } else {
        T s = (weights[i] < 0.0) ? -1.0 : 1.0;
        T b = log2(abs(weights[i]));
        T d = 0.58496250072115619; // quantization threshold log2(1.5)
        // (d = 0.5: use geometric mean; d = log2(1.5): use arithmetic mean)
        int e = (int)(floor(b) + (b - floor(b) >= d));

        if (e > n1) {
          e = n1;
        }
        if (e < n2) {
          e = n2;
        }
        weights[i] = ldexp(s, e);
      }
    }
  }
}

template <typename T, typename T1>
struct cmp : public thrust::binary_function<int, int, bool> {
  cmp(const T *ptr) : weights(ptr) {}

  __host__ __device__ bool operator()(const int i, const int j) const {
    return abs(weights[i]) > abs(weights[j]);
  }

  const T *weights;
};

template <typename T>
struct absolute_value : public thrust::unary_function<T, T> {
  __host__ __device__ T operator()(const T &x) const {
    return x < T(0) ? -x : x;
  }
};

template <typename T, typename T1>
__global__ void kernel_compute_learnable(const int size,
                                         int *cumulative_count) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    cumulative_count[i] = i + 1 - cumulative_count[i];
  }
}

template <typename T, typename T1>
void INQAffineCuda<T, T1>::setup_impl(const Variables &inputs,
                                      const Variables &outputs) {
  cuda_set_device(this->device_);

  // A: Check shape of indicator matrix
  NBLA_CHECK(inputs[1]->shape().size() == inputs[2]->shape().size(),
             error_code::value,
             "Indicators and weights must have same size. "
             "Ndim of weights: %d != ndim of indicators: %d.",
             inputs[1]->shape().size(), inputs[2]->shape().size());
  for (int i = 0; i < inputs[1]->shape().size(); ++i) {
    NBLA_CHECK(inputs[1]->shape()[i] == inputs[2]->shape()[i],
               error_code::value,
               "Indicators and weights must have same size. "
               "weight shape[%d]: %d != indicator shape[%d]: %d.",
               i, inputs[1]->shape()[i], i, inputs[2]->shape()[i]);
  }

  // B: Check that chosen algorithm is either "largest_abs" or "random"
  NBLA_CHECK(this->selection_algorithm_ == "largest_abs" ||
                 this->selection_algorithm_ == "random",
             error_code::value,
             "Provided value for selection algorithm not valid: %s."
             "Valid values are \"largest_abs\" and \"random\".",
             this->selection_algorithm_.c_str());

  // C: Initialize internal `affine` function
  this->affine_ = create_Affine(this->ctx_, this->base_axis_);
  if (inputs.size() == 4) { // with bias
    this->affine_->setup(Variables{inputs[0], inputs[1], inputs[3]}, outputs);
  } else { // without bias
    this->affine_->setup(Variables{inputs[0], inputs[1]}, outputs);
  }

  // D: Initialize random number generator (required for randomly selecting the
  // indices to fix)
  if (this->selection_algorithm_ == "random") {
    // if seed is set, create local curand generator.
    if (this->seed_ != -1) {
      // CURAND_RNG_PSEUDO_DEFAULT is CURAND_RNG_PSEUDO_XORWOW.
      curand_generator_ = curand_create_generator(this->seed_);
    }
  }

  // F: Initialize minibatch counter and internal copies of weights/indicators
  this->minibatch_counter_ = 0;
  this->old_weights_.reshape(inputs[1]->shape(), true);
  this->old_indicators_.reshape(inputs[1]->shape(), true);
  this->old_indicators_.data()->zero();

  // G: Initialize temporary memory for INQ (only CUDA implementation)
  this->indices_.reshape(inputs[1]->shape(), true);
  this->cumulative_count_.reshape(inputs[1]->shape(), true);
}

template <typename T, typename T1>
void INQAffineCuda<T, T1>::forward_impl(const Variables &inputs,
                                        const Variables &outputs) {
  typedef typename CudaTypeForceFloat<T>::type Tc;
  cuda_set_device(this->device_);

  Tc *weights = inputs[1]->cast_data_and_get_pointer<Tc>(this->ctx_);
  Tc *old_weights = ((Variable &)this->old_weights_)
                        .cast_data_and_get_pointer<Tc>(this->ctx_);

  T1 *indicators = inputs[2]->cast_data_and_get_pointer<T1>(this->ctx_);
  T1 *old_indicators = ((Variable &)this->old_indicators_)
                           .cast_data_and_get_pointer<T1>(this->ctx_);

  int *indices =
      ((Variable &)this->indices_).cast_data_and_get_pointer<int>(this->ctx_);
  int *cumulative_count = ((Variable &)this->cumulative_count_)
                              .cast_data_and_get_pointer<int>(this->ctx_);

  // A: Go through each element and copy old value for weights if weight was
  // fixed before.
  //    This is done to make sure that we do not update fixed weights.
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_copy_fixedweights<Tc, T1>),
                                 inputs[1]->size(), old_weights, old_indicators,
                                 weights);

  // B: Check whether we need to fix 50% of the learnable weights
  //    Note: The following code block makes usage of Thrust. As it is only
  //    called a few times per training
  //          this should be not a performance bottleneck. TODO: Replace Thrust
  //          code by custom CUDA kernels
  if (std::find(this->inq_iterations_.begin(), this->inq_iterations_.end(),
                this->minibatch_counter_) != this->inq_iterations_.end()) {
    // determine weights that we will fix
    if (this->inq_iterations_.back() == this->minibatch_counter_) {
      // if we have reached the last element in `inq_iterations` then we fix all
      // weights
      thrust::fill(thrust::device_pointer_cast<T1>(indicators),
                   thrust::device_pointer_cast<T1>(indicators) +
                       inputs[1]->size(),
                   (T1)1);
    } else {
      // not last element in `inq_iterations`, hence fix 50% of the learnable
      // weights
      if (this->selection_algorithm_ == "largest_abs") {
        // fix weights with largest absolute value
        thrust::sequence(
            thrust::device_pointer_cast<int>(indices),
            thrust::device_pointer_cast<int>(indices) + inputs[1]->size(), 0);
        thrust::sort(thrust::device_pointer_cast<int>(indices),
                     thrust::device_pointer_cast<int>(indices) +
                         inputs[1]->size(),
                     cmp<Tc, T1>(thrust::raw_pointer_cast(weights)));

        int num_learnable =
            inputs[1]->size() -
            thrust::reduce(thrust::device_pointer_cast<T1>(indicators),
                           thrust::device_pointer_cast<T1>(indicators) +
                               inputs[1]->size(),
                           0, thrust::plus<int>());

        thrust::inclusive_scan(
            thrust::make_permutation_iterator(
                thrust::device_pointer_cast<T1>(indicators),
                thrust::device_pointer_cast<int>(indices)),
            thrust::make_permutation_iterator(
                thrust::device_pointer_cast<T1>(indicators),
                thrust::device_pointer_cast<int>(indices) + inputs[1]->size()),
            thrust::device_pointer_cast<T1>(cumulative_count));
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
            (kernel_compute_learnable<Tc, T1>), // turns number of fixed weights
                                                // into number of learnable
                                                // weights
            inputs[1]->size(), cumulative_count);

        int idx =
            thrust::find(thrust::device_pointer_cast<int>(cumulative_count),
                         thrust::device_pointer_cast<int>(cumulative_count) +
                             inputs[1]->size(),
                         num_learnable / 2) -
            thrust::device_pointer_cast<int>(cumulative_count) + 1;
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_fix_largestabs<Tc, T1>), idx,
                                       indicators, indices);
      } else {
        // random selection (we re-use old_weights here to keep the random
        // values)
        curandGenerator_t &gen =
            this->seed_ == -1
                ? SingletonManager::get<Cuda>()->curand_generator()
                : curand_generator_;
        curand_generate_rand<Tc>(gen, 0.0f, 1.0f, old_weights,
                                 inputs[0]->size());
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_random_selection<Tc, T1>),
                                       inputs[1]->size(), indicators,
                                       old_weights);
      }
    }
  }

  // C: convert all fixed weights to power-of-two values
  Tc max_absval = thrust::transform_reduce(
      thrust::device_pointer_cast<Tc>(weights),
      thrust::device_pointer_cast<Tc>(weights) + inputs[1]->size(),
      absolute_value<Tc>(), 0.0, thrust::maximum<Tc>());

  if (max_absval == 0.0f) {
    max_absval = 1.0f;
  }
  int n1 = (int)(std::floor(std::log2(max_absval)) +
                 (std::log2(max_absval) - std::floor(std::log2(max_absval)) >=
                  std::log2(1.5)));
  int n2 = n1 + 1 - (int)std::pow(2, this->num_bits_ - 2);
  Tc pruning_threshold = std::pow(2, n2 - 1);

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_quantize_weights<Tc, T1>),
                                 inputs[1]->size(), indicators, weights, n1, n2,
                                 pruning_threshold);

  // D: Calculate the forward pass
  if (inputs.size() == 4) { // with bias
    this->affine_->forward(Variables{inputs[0], inputs[1], inputs[3]}, outputs);
  } else {
    this->affine_->forward(Variables{inputs[0], inputs[1]}, outputs);
  }

  // E: Increase minibatch counter
  this->minibatch_counter_++;

  // F: Store weights/indicators
  cudaMemcpy(old_weights, weights, inputs[1]->size() * sizeof(Tc),
             cudaMemcpyDeviceToDevice);
  cudaMemcpy(old_indicators, indicators, inputs[1]->size() * sizeof(T1),
             cudaMemcpyDeviceToDevice);
}

template <typename T, typename T1>
void INQAffineCuda<T, T1>::backward_impl(const Variables &inputs,
                                         const Variables &outputs,
                                         const vector<bool> &propagate_down,
                                         const vector<bool> &accum) {
  INQAffine<T, T1>::backward_impl(inputs, outputs, propagate_down, accum);
}
}
