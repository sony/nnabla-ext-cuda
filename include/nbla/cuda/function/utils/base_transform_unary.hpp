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
#ifndef __NBLA_CUDA_FUNCTION_BASE_TRANSFORM_UNARY_HPP__
#define __NBLA_CUDA_FUNCTION_BASE_TRANSFORM_UNARY_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/function.hpp>
#include <nbla/function/utils/base_transform_unary.hpp>

#include <tuple>

namespace nbla {

template <typename T, typename... Args>
class TransformUnaryCuda : public BaseTransformUnary<Args...> {
protected:
  int device_;

public:
  TransformUnaryCuda(const Context &ctx, Args... args)
      : BaseTransformUnary<Args...>(ctx, args...),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~TransformUnaryCuda() {}
  virtual vector<dtypes> in_types() {
    return vector<dtypes>{get_dtype<T>(), get_dtype<T>()};
  }
  virtual vector<dtypes> out_types() { return vector<dtypes>{get_dtype<T>()}; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }
};

#define NBLA_DECLARE_TRANSFORM_UNARY_CUDA_CLASS_COMMON(NAME)                   \
public:                                                                        \
  virtual ~NAME##Cuda() {}                                                     \
  virtual string name() { return #NAME "Cuda"; }

#define NBLA_DECLARE_TRANSFORM_UNARY_CUDA_FORWARD_BACKWARD()                   \
protected:                                                                     \
  virtual void forward_impl(const Variables &inputs,                           \
                            const Variables &outputs);                         \
  virtual void backward_impl(                                                  \
      const Variables &inputs, const Variables &outputs,                       \
      const vector<bool> &propagate_down, const vector<bool> &accum)

// ----------------------------------------------------------------------------
// Zero argument
// ----------------------------------------------------------------------------
#define NBLA_DECLARE_TRANSFORM_UNARY_CUDA_CLASS_BEGIN(NAME)                    \
  template <typename T> class NAME##Cuda : public TransformUnaryCuda<T>

#define NBLA_DECLARE_TRANSFORM_UNARY_CUDA(NAME)                                \
  NBLA_DECLARE_TRANSFORM_UNARY_CUDA_CLASS_BEGIN(NAME) {                        \
    NBLA_DECLARE_TRANSFORM_UNARY_CUDA_CLASS_COMMON(NAME);                      \
    explicit NAME##Cuda(const Context &ctx) : TransformUnaryCuda<T>(ctx) {}    \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_);                                        \
    }                                                                          \
    NBLA_DECLARE_TRANSFORM_UNARY_CUDA_FORWARD_BACKWARD();                      \
                                                                               \
  public:                                                                      \
    virtual bool grad_depends_output_data(int i, int o) const;                 \
  }

// ----------------------------------------------------------------------------
// One argument
// ----------------------------------------------------------------------------
#define NBLA_DECLARE_TRANSFORM_UNARY_CUDA_CLASS_BEGIN_N(NAME, ...)             \
  template <typename T>                                                        \
  class NAME##Cuda : public TransformUnaryCuda<T, __VA_ARGS__>

#define NBLA_DECLARE_TRANSFORM_UNARY_CUDA_1(NAME, A0)                          \
  NBLA_DECLARE_TRANSFORM_UNARY_CUDA_CLASS_BEGIN_N(NAME, A0) {                  \
    NBLA_DECLARE_TRANSFORM_UNARY_CUDA_CLASS_COMMON(NAME);                      \
    explicit NAME##Cuda(const Context &ctx, const A0 &a0)                      \
        : TransformUnaryCuda<T, A0>(ctx, a0) {}                                \
    virtual shared_ptr<Function> copy() const {                                \
      return create_##NAME(this->ctx_, std::get<0>(this->args_));              \
    }                                                                          \
    NBLA_DECLARE_TRANSFORM_UNARY_CUDA_FORWARD_BACKWARD();                      \
                                                                               \
  public:                                                                      \
    virtual bool grad_depends_output_data(int i, int o) const;                 \
  }
}
#endif
