// Copyright 2018,2019,2020,2021 Sony Corporation.
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
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/broadcast.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/function/transpose.hpp>

#include <numeric>

namespace nbla {

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
template <typename T>
void BroadcastCuda<T>::setup_impl(const Variables &inputs,

                                  const Variables &outputs) {
  Broadcast<T>::setup_impl(inputs, outputs);
  int ndim = outputs[0]->ndim();
  auto eshape = expand_shape(inputs[0]->shape(), ndim);
  vector<int> broadcast_dims;
  if (inputs[0]->ndim() == 0) {
    // If input is a scalar.
    broadcast_dims.resize(this->shape_.size());
    std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);
  } else {
    for (int d = 0; d < ndim; ++d) {
      if (this->shape_[d] != eshape[d])
        broadcast_dims.push_back(d);
    }
  }
  broadcast_dims_ = broadcast_dims;
  if (broadcast_dims.size() == 0)
    return;
  f_sum_ = create_Sum(this->ctx_, /*axis*/ broadcast_dims, /*keepdims*/ true);
}

// ----------------------------------------------------------------------------
// Strided index getter
// ----------------------------------------------------------------------------
template <int ND> struct strided_index_cuda {
  static __device__ int get(int y_index, const int *stride_x,
                            const int *shape_y) {
    int stride = 1;
    int x_index = 0;
    strided_index_cuda<ND - 1>::_get(y_index, stride_x, shape_y, stride,
                                     x_index);
    return x_index;
  }
  static __device__ void _get(int y_index, const int *stride_x,
                              const int *shape_y, int &stride, int &x_index) {
    const int dim_index = int(y_index / stride) % shape_y[ND];
    stride *= shape_y[ND];
    x_index += dim_index * stride_x[ND];
    strided_index_cuda<ND - 1>::_get(y_index, stride_x, shape_y, stride,
                                     x_index);
  }
};
template <> struct strided_index_cuda<0> {
  static __device__ int get(int y_index, const int *stride_x,
                            const int *shape_y) {
    return 0;
  }
  static __device__ void _get(int y_index, const int *stride_x,
                              const int *shape_y, int &stride, int &x_index) {
    const int dim_index = int(y_index / stride) % shape_y[0];
    stride *= shape_y[0];
    x_index += dim_index * stride_x[0];
  }
};

// ----------------------------------------------------------------------------
// Broadcast kernel
// ----------------------------------------------------------------------------
template <int Ndim, typename T>
__global__ void kernel_broadcast(size_t size, const T *x, const int *stride_x,
                                 const int *shape_y, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    int jdx = strided_index_cuda<Ndim>::get(idx, stride_x, shape_y);
    y[idx] = x[jdx];
  }
}

// ----------------------------------------------------------------------------
// Unrolled broadcast caller for templated dimension
// ----------------------------------------------------------------------------
template <int ND, typename T> struct switch_broadcast_cuda {
  static void call(int num, size_t size, const T *x, const int *stride_x,
                   const int *shape_y, T *y) {
    if (ND == num) {
      const int blocks = NBLA_CUDA_GET_BLOCKS(size);
      const int inkernel_loop = NBLA_CEIL_INT_DIV(blocks, NBLA_CUDA_MAX_BLOCKS);
      const int total_blocks = NBLA_CEIL_INT_DIV(blocks, inkernel_loop);
      kernel_broadcast<ND, T><<<total_blocks, NBLA_CUDA_NUM_THREADS>>>(
          size, x, stride_x, shape_y, y);
      NBLA_CUDA_KERNEL_CHECK();
      return;
    }
    switch_broadcast_cuda<ND - 1, T>::call(num, size, x, stride_x, shape_y, y);
  }
};

template <typename T> struct switch_broadcast_cuda<-1, T> {
  static void call(int num, size_t size, const T *x, const int *stride_x,
                   const int *shape_y, T *y) {
    NBLA_ERROR(error_code::not_implemented,
               "Broadcast is not implemented for %d dimensional array.", num);
  }
};

// ----------------------------------------------------------------------------
// Forward
// ----------------------------------------------------------------------------
template <typename T>
void BroadcastCuda<T>::forward_impl(const Variables &inputs,
                                    const Variables &outputs) {
  const Tc *x = inputs[0]->get_data_pointer<Tc>(this->ctx_);
  Tc *y = outputs[0]->cast_data_and_get_pointer<Tc>(this->ctx_, true);
  auto _iarr = [this](Variable &v) {
    return v.get_data_pointer<int>(this->ctx_);
  };
  const int *stride_x = _iarr(this->stride_x_);
  // for (int i = 0; i < this->stride_x_.ndim(); i++) {
  //   printf("stride_x[%d] = %d\n", i, stride_x[i]);
  // }
  const int *shape_y = _iarr(this->shape_y_);
  int ndim = outputs[0]->ndim();
  int size = outputs[0]->size();
  cuda_set_device(device_);
  switch_broadcast_cuda<NBLA_BROADCAST_MAX_DIM, Tc>::call(ndim, size, x,
                                                          stride_x, shape_y, y);
}

template <typename T>
__global__ void kernel_add_grad(int size, const T *g, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { dx[idx] += g[idx]; }
}

template <typename T>
void BroadcastCuda<T>::backward_impl(const Variables &inputs,
                                     const Variables &outputs,
                                     const vector<bool> &propagate_down,
                                     const vector<bool> &accum) {
  if (!propagate_down[0])
    return;
  shared_ptr<Variable> sum_input = make_shared<Variable>(outputs[0]->grad());
  shared_ptr<Variable> sum_output;
  if (f_sum_) {
    if (!accum[0]) {
      sum_output = make_shared<Variable>(inputs[0]->grad());
      f_sum_->setup(Variables{sum_input.get()}, Variables{sum_output.get()});
      f_sum_->forward(Variables{sum_input.get()}, Variables{sum_output.get()});
      return;
    }
    sum_output = make_shared<Variable>(inputs[0]->shape());
    f_sum_->setup(Variables{sum_input.get()}, Variables{sum_output.get()});
    f_sum_->forward(Variables{sum_input.get()}, Variables{sum_output.get()});
  } else {
    if (!accum[0])
      inputs[0]->grad()->zero();
  }
  auto _get = [this](Variable *v) {
    return v->get_data_pointer<Tc>(this->ctx_);
  };
  auto _gget = [this](Variable *v) {
    return v->get_grad_pointer<Tc>(this->ctx_);
  };
  cuda_set_device(device_);
  const Tc *g = f_sum_ ? _get(sum_output.get()) : _gget(outputs[0]);
  Tc *dx = inputs[0]->cast_grad_and_get_pointer<Tc>(this->ctx_, false);
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_add_grad, inputs[0]->size(), g, dx);
}
} // namespace nbla
