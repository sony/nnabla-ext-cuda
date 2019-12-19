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

/** Base class of unary operations for CUDA.
 */
#ifndef __NBLA_CUDA_FUNCTION_BASE_TRANSFORM_UNARY_CUH__
#define __NBLA_CUDA_FUNCTION_BASE_TRANSFORM_UNARY_CUH__

#include <nbla/cuda/function/utils/base_transform_unary.hpp>

#include <tuple>

namespace nbla {
using std::tuple;

class BaseUnaryOpCuda {
public:
  template <typename T>
  __forceinline__ __device__ T operator()(const T x0, const T x1) {
    return 0;
  }
  template <typename T>
  __forceinline__ __device__ T g(const T dy, const T x, const T y) {
    return 0;
  }
  __host__ void verify_g() {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 0 is not implemented.");
  }
};

template <typename T, typename UnaryOp>
__global__ void kernel_transform_unary(int size, const T *x, T *y, UnaryOp op) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = op(x[idx]); }
}

template <typename T, typename UnaryOp, bool accum>
__global__ void kernel_transform_unary_grad(int size, const T *dy, const T *x,
                                            const T *y, T *g, UnaryOp op) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    g[idx] = (accum ? g[idx] : (T)0) + op.g(dy[idx], x[idx], y[idx]);
  }
}

template <typename T, typename UnaryOp>
void forward_impl_transform_unary(const Variables &inputs,
                                  const Variables &outputs, Context &ctx,
                                  UnaryOp op) {
  cuda_set_device(std::stoi(ctx.device_id));
  const T *x = inputs[0]->get_data_pointer<T>(ctx);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(ctx, true);
  int size = inputs[0]->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_transform_unary, size, x, y, op);
}

template <class T, typename UnaryOp>
void backward_impl_transform_unary(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum, Context &ctx,
                                   UnaryOp op) {
  if (!propagate_down[0]) {
    return;
  }
  op.verify_g();
  cuda_set_device(std::stoi(ctx.device_id));
  const T *dy = outputs[0]->get_grad_pointer<T>(ctx);
  const T *x = inputs[0]->get_data_pointer<T>(ctx);
  const T *y = outputs[0]->get_data_pointer<T>(ctx);
  size_t size = inputs[0]->size();
  T *g = inputs[0]->cast_grad_and_get_pointer<T>(ctx, !accum[0]);
  if (accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_transform_unary_grad<T, UnaryOp, true>), size, dy, x, y, g, op);
  } else {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_transform_unary_grad<T, UnaryOp, false>), size, dy, x, y, g,
        op);
  }
}

#define NBLA_DEFINE_UNARY_OP_CUDA_FORWARD(OP)                                  \
  template <typename T> __forceinline__ __device__ T operator()(const T x) {   \
    return OP;                                                                 \
  }

#define NBLA_DEFINE_UNARY_OP_CUDA_BACKWARD(GOP)                                \
  template <typename T>                                                        \
  __forceinline__ __device__ T g(const T dy, const T x, const T y) {           \
    return GOP;                                                                \
  }                                                                            \
  __host__ void verify_g() {}

#define NBLA_DEFINE_TRANSFORM_UNARY_CUDA_FORWARD_BACKWARD(NAME)                \
  template <typename T>                                                        \
  void NAME##Cuda<T>::forward_impl(const Variables &inputs,                    \
                                   const Variables &outputs) {                 \
    forward_impl_transform_unary<typename CudaType<T>::type>(                  \
        inputs, outputs, this->ctx_, NAME##UnaryOpCuda(this->args_));          \
  }                                                                            \
  template <typename T>                                                        \
  void NAME##Cuda<T>::backward_impl(                                           \
      const Variables &inputs, const Variables &outputs,                       \
      const vector<bool> &propagate_down, const vector<bool> &accum) {         \
    backward_impl_transform_unary<typename CudaType<T>::type>(                 \
        inputs, outputs, propagate_down, accum, this->ctx_,                    \
        NAME##UnaryOpCuda(this->args_));                                       \
  }

#define NBLA_DEFINE_GRAD_DEPENDS_OUTPUT_DATA(NAME, DEP_Y)                      \
  template <typename T>                                                        \
  bool NAME##Cuda<T>::grad_depends_output_data(int i, int o) const {           \
    return DEP_Y;                                                              \
  };

// ----------------------------------------------------------------------------
// Zero argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_UNARY_OP_CUDA_NO_GRAD(NAME, OP)                            \
  class NAME##UnaryOpCuda : public BaseUnaryOpCuda {                           \
  public:                                                                      \
    __inline__ NAME##UnaryOpCuda(const tuple<> &dummy) {}                      \
    NBLA_DEFINE_UNARY_OP_CUDA_FORWARD(OP)                                      \
  }

#define NBLA_DEFINE_UNARY_OP_CUDA(NAME, OP, GOP)                               \
  class NAME##UnaryOpCuda : public BaseUnaryOpCuda {                           \
  public:                                                                      \
    __inline__ NAME##UnaryOpCuda(const tuple<> &dummy) {}                      \
    NBLA_DEFINE_UNARY_OP_CUDA_FORWARD(OP)                                      \
    NBLA_DEFINE_UNARY_OP_CUDA_BACKWARD(GOP)                                    \
  }

#define NBLA_DEFINE_TRANSFORM_UNARY_CUDA_NO_GRAD(NAME, OP)                     \
  NBLA_DEFINE_UNARY_OP_CUDA_NO_GRAD(NAME, OP);                                 \
  NBLA_DEFINE_TRANSFORM_UNARY_CUDA_FORWARD_BACKWARD(NAME)                      \
  NBLA_DEFINE_GRAD_DEPENDS_OUTPUT_DATA(NAME, false)

#define NBLA_DEFINE_TRANSFORM_UNARY_CUDA(NAME, OP, GOP, DEP_Y)                 \
  NBLA_DEFINE_UNARY_OP_CUDA(NAME, OP, GOP);                                    \
  NBLA_DEFINE_TRANSFORM_UNARY_CUDA_FORWARD_BACKWARD(NAME)                      \
  NBLA_DEFINE_GRAD_DEPENDS_OUTPUT_DATA(NAME, DEP_Y)

// ----------------------------------------------------------------------------
// One argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_UNARY_OP_CUDA_1(NAME, OP, GOP, A0)                         \
  class NAME##UnaryOpCuda : public BaseUnaryOpCuda {                           \
  public:                                                                      \
    A0 a0;                                                                     \
    __inline__ NAME##UnaryOpCuda(const tuple<A0> &args)                        \
        : BaseUnaryOpCuda(), a0(std::get<0>(args)) {}                          \
    NBLA_DEFINE_UNARY_OP_CUDA_FORWARD(OP)                                      \
    NBLA_DEFINE_UNARY_OP_CUDA_BACKWARD(GOP)                                    \
  }

#define NBLA_DEFINE_TRANSFORM_UNARY_CUDA_1(NAME, OP, GOP, A0, DEP_Y)           \
  NBLA_DEFINE_UNARY_OP_CUDA_1(NAME, OP, GOP, A0);                              \
  NBLA_DEFINE_TRANSFORM_UNARY_CUDA_FORWARD_BACKWARD(NAME)                      \
  NBLA_DEFINE_GRAD_DEPENDS_OUTPUT_DATA(NAME, DEP_Y)

#define NBLA_DEFINE_UNARY_OP_CUDA_1_NO_GRAD(NAME, OP, A0)                      \
  class NAME##UnaryOpCuda : public BaseUnaryOpCuda {                           \
  public:                                                                      \
    A0 a0;                                                                     \
    __inline__ NAME##UnaryOpCuda(const tuple<A0> &args)                        \
        : BaseUnaryOpCuda(), a0(std::get<0>(args)) {}                          \
    NBLA_DEFINE_UNARY_OP_CUDA_FORWARD(OP)                                      \
  }

#define NBLA_DEFINE_TRANSFORM_UNARY_CUDA_1_NO_GRAD(NAME, OP, A0)               \
  NBLA_DEFINE_UNARY_OP_CUDA_1_NO_GRAD(NAME, OP, A0);                           \
  NBLA_DEFINE_TRANSFORM_UNARY_CUDA_FORWARD_BACKWARD(NAME)                      \
  NBLA_DEFINE_GRAD_DEPENDS_OUTPUT_DATA(NAME, false)
}
#endif
