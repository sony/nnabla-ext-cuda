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

/** Base class of binary operations for CUDA.
 */
#ifndef __NBLA_CUDA_FUNCTION_BASE_TRANSFORM_BINARY_CUH__
#define __NBLA_CUDA_FUNCTION_BASE_TRANSFORM_BINARY_CUH__

#include <nbla/cuda/function/utils/base_transform_binary.hpp>

#include <tuple>

namespace nbla {
using std::tuple;

class BaseBinaryOpCuda {
public:
  template <typename T>
  __forceinline__ __device__ T operator()(const T x0, const T x1) {
    return 0;
  }
  template <typename T>
  __forceinline__ __device__ T g0(const T dy, const T x0, const T x1,
                                  const T y) {
    return 0;
  }
  template <typename T>
  __forceinline__ __device__ T g1(const T dy, const T x0, const T x1,
                                  const T y) {
    return 0;
  }
  __host__ void verify_g0() {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 0 is not implemented.");
  }
  __host__ void verify_g1() {
    NBLA_ERROR(error_code::not_implemented,
               "Backward operation for input 1 is not implemented.");
  }
};

template <typename T, typename BinaryOp>
__global__ void kernel_transform_binary(int size, const T *x0, const T *x1,
                                        T *y, BinaryOp op) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { y[idx] = op(x0[idx], x1[idx]); }
}

template <typename T, typename BinaryOp, bool accum>
__global__ void kernel_transform_binary_grad0(int size, const T *dy,
                                              const T *x0, const T *x1,
                                              const T *y, T *g0, BinaryOp op) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    g0[idx] =
        (accum ? g0[idx] : (T)0) + op.g0(dy[idx], x0[idx], x1[idx], y[idx]);
  }
}

template <typename T, typename BinaryOp, bool accum>
__global__ void kernel_transform_binary_grad1(int size, const T *dy,
                                              const T *x0, const T *x1,
                                              const T *y, T *g1, BinaryOp op) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    g1[idx] =
        (accum ? g1[idx] : (T)0) + op.g1(dy[idx], x0[idx], x1[idx], y[idx]);
  }
}

template <typename T, typename BinaryOp>
void forward_impl_transform_binary(const Variables &inputs,
                                   const Variables &outputs, Context &ctx,
                                   Function *f_bc0, Variable *o_bc0,
                                   Function *f_bc1, Variable *o_bc1,
                                   BinaryOp op) {
  auto _get = [&ctx](Variable *v) { return v->get_data_pointer<T>(ctx); };
  if (f_bc0) {
    f_bc0->forward(Variables{inputs[0]}, Variables{o_bc0});
  }
  if (f_bc1) {
    f_bc1->forward(Variables{inputs[1]}, Variables{o_bc1});
  }
  const T *x0 = _get(f_bc0 ? o_bc0 : inputs[0]);
  const T *x1 = _get(f_bc1 ? o_bc1 : inputs[1]);
  T *y = outputs[0]->cast_data_and_get_pointer<T>(ctx, true);
  int size = outputs[0]->size();
  cuda_set_device(std::stoi(ctx.device_id));
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_transform_binary, size, x0, x1, y, op);
}

template <class T, typename BinaryOp>
void backward_impl_transform_binary(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum, Context &ctx,
                                    Function *f_bc0, Variable *o_bc0,
                                    Function *f_bc1, Variable *o_bc1,
                                    BinaryOp op) {
  if (!(propagate_down[0] || propagate_down[1]))
    return;
  cuda_set_device(std::stoi(ctx.device_id));
  auto _get_data = [&ctx](Variable *v) { return v->get_data_pointer<T>(ctx); };
  auto _cast_grad = [&ctx](Variable *v, bool wo) {
    return v->cast_grad_and_get_pointer<T>(ctx, wo);
  };
  const T *dy = outputs[0]->get_grad_pointer<T>(ctx);
  const T *x0 = _get_data(f_bc0 ? o_bc0 : inputs[0]);
  const T *x1 = _get_data(f_bc1 ? o_bc1 : inputs[1]);
  const T *y = _get_data(outputs[0]);
  size_t size = outputs[0]->size();
  if (propagate_down[0]) {
    op.verify_g0();
    T *dx0 = f_bc0 ? _cast_grad(o_bc0, true) : _cast_grad(inputs[0], !accum[0]);
    if ((!f_bc0) && accum[0]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_transform_binary_grad0<T, BinaryOp, true>), size, dy, x0, x1,
          y, dx0, op);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_transform_binary_grad0<T, BinaryOp, false>), size, dy, x0, x1,
          y, dx0, op);
    }
    if (f_bc0) {
      f_bc0->backward(Variables{inputs[0]}, Variables{o_bc0}, {true},
                      {accum[0]});
    }
  }
  if (propagate_down[1]) {
    op.verify_g1();
    T *dx1 = f_bc1 ? _cast_grad(o_bc1, true) : _cast_grad(inputs[1], !accum[1]);
    if ((!f_bc1) && accum[1]) {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_transform_binary_grad1<T, BinaryOp, true>), size, dy, x0, x1,
          y, dx1, op);
    } else {
      NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
          (kernel_transform_binary_grad1<T, BinaryOp, false>), size, dy, x0, x1,
          y, dx1, op);
    }
    if (f_bc1) {
      f_bc1->backward(Variables{inputs[1]}, Variables{o_bc1}, {true},
                      {accum[1]});
    }
  }
}

#define NBLA_DEFINE_BINARY_OP_CUDA_CLASS(NAME)                                 \
  class NAME##BinaryOpCuda : public BaseBinaryOpCuda

#define NBLA_DEFINE_BINARY_OP_CUDA_FORWARD(OP)                                 \
  template <typename T>                                                        \
  __forceinline__ __device__ T operator()(const T x0, const T x1) {            \
    return OP;                                                                 \
  }

#define NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(NUM, GOP)                          \
  template <typename T>                                                        \
  __forceinline__ __device__ T g##NUM(const T dy, const T x0, const T x1,      \
                                      const T y) {                             \
    return GOP;                                                                \
  }                                                                            \
  __host__ void verify_g##NUM() {}

#define NBLA_DEFINE_TRANSFORM_BINARY_CUDA_FORWARD_BACKWARD(NAME)               \
  template <typename T>                                                        \
  void NAME##Cuda<T>::forward_impl(const Variables &inputs,                    \
                                   const Variables &outputs) {                 \
    forward_impl_transform_binary<typename CudaType<T>::type>(                 \
        inputs, outputs, this->ctx_, this->f_bc0_.get(), this->o_bc0_.get(),   \
        this->f_bc1_.get(), this->o_bc1_.get(),                                \
        NAME##BinaryOpCuda(this->args_));                                      \
  }                                                                            \
                                                                               \
  template <typename T>                                                        \
  void NAME##Cuda<T>::backward_impl(                                           \
      const Variables &inputs, const Variables &outputs,                       \
      const vector<bool> &propagate_down, const vector<bool> &accum) {         \
    backward_impl_transform_binary<typename CudaType<T>::type>(                \
        inputs, outputs, propagate_down, accum, this->ctx_,                    \
        this->f_bc0_.get(), this->o_bc0_.get(), this->f_bc1_.get(),            \
        this->o_bc1_.get(), NAME##BinaryOpCuda(this->args_));                  \
  }

// ----------------------------------------------------------------------------
// Zero argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_BINARY_OP_CUDA_NO_GRAD(NAME, OP)                           \
  NBLA_DEFINE_BINARY_OP_CUDA_CLASS(NAME) {                                     \
  public:                                                                      \
    __inline__ __host__ __device__ NAME##BinaryOpCuda(const tuple<> &dummy) {} \
    NBLA_DEFINE_BINARY_OP_CUDA_FORWARD(OP)                                     \
  }

#define NBLA_DEFINE_BINARY_OP_CUDA(NAME, OP, GOP0, GOP1)                       \
  NBLA_DEFINE_BINARY_OP_CUDA_CLASS(NAME) {                                     \
  public:                                                                      \
    __inline__ __host__ __device__ NAME##BinaryOpCuda(const tuple<> &dummy) {} \
    NBLA_DEFINE_BINARY_OP_CUDA_FORWARD(OP)                                     \
    NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(0, GOP0)                               \
    NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(1, GOP1)                               \
  }
#define NBLA_DEFINE_TRANSFORM_BINARY_CUDA_NO_GRAD(NAME, OP)                    \
  NBLA_DEFINE_BINARY_OP_CUDA_NO_GRAD(NAME, OP);                                \
  NBLA_DEFINE_TRANSFORM_BINARY_CUDA_FORWARD_BACKWARD(NAME)

#define NBLA_DEFINE_TRANSFORM_BINARY_CUDA(NAME, OP, GOP0, GOP1)                \
  NBLA_DEFINE_BINARY_OP_CUDA(NAME, OP, GOP0, GOP1);                            \
  NBLA_DEFINE_TRANSFORM_BINARY_CUDA_FORWARD_BACKWARD(NAME)

// ----------------------------------------------------------------------------
// One argument
// ----------------------------------------------------------------------------
#define NBLA_DEFINE_BINARY_OP_CUDA_1(NAME, OP, GOP0, GOP1, A0)                 \
  NBLA_DEFINE_BINARY_OP_CUDA_CLASS(NAME) {                                     \
  public:                                                                      \
    A0 a0;                                                                     \
    __inline__ NAME##BinaryOpCuda(const tuple<A0> &args)                       \
        : a0(std::get<0>(args)) {}                                             \
    NBLA_DEFINE_BINARY_OP_CUDA_FORWARD(OP)                                     \
    NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(0, GOP0)                               \
    NBLA_DEFINE_BINARY_OP_CUDA_BACKWARD(1, GOP1)                               \
  }
#define NBLA_DEFINE_TRANSFORM_BINARY_CUDA_1(NAME, OP, GOP0, GOP1, A0)          \
  NBLA_DEFINE_BINARY_OP_CUDA_1(NAME, OP, GOP0, GOP1, A0);                      \
  NBLA_DEFINE_TRANSFORM_BINARY_CUDA_FORWARD_BACKWARD(NAME)
}
#endif
