// Copyright 2019,2020,2021 Sony Corporation.
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

#ifndef NBLA_CUDA_FUNCTION_GENERIC_CLIP_GRAD_CUH
#define NBLA_CUDA_FUNCTION_GENERIC_CLIP_GRAD_CUH

#include <nbla/cuda/common.hpp>
#include <nbla/function/pow_scalar.hpp>
#include <nbla/function/sum.hpp>
#include <nbla/variable.hpp>

namespace nbla {
template <typename T>
__global__ void kernel_clip_grad_by_norm(const int num, T *grad, const T *l2sum,
                                         const float clip_norm) {
  // to avoid zero division
  if (*l2sum == 0.0 || *l2sum <= clip_norm * clip_norm)
    return;
  const float norm = sqrtf(*l2sum);
  NBLA_CUDA_KERNEL_LOOP(idx, num) { grad[idx] = clip_norm * grad[idx] / norm; }
}

template <typename T>
void clip_grad_by_norm_cuda(const Context &ctx,
                            const shared_ptr<Variable> param, float clip_norm) {
  cuda_set_device(std::stoi(ctx.device_id));

  Variable g(param->grad());
  Variable g_pow(param->shape());
  Variable sum(Shape_t{});

  // calculate g^2
  auto f_pow_scalar = create_PowScalar(ctx, 2.0, false);
  f_pow_scalar->setup(Variables{&g}, Variables{&g_pow});
  f_pow_scalar->forward(Variables{&g}, Variables{&g_pow});

  // sum all gradients
  vector<int> axis;
  for (int i = 0; i < param->ndim(); ++i)
    axis.push_back(i);
  auto f_sum = create_Sum(ctx, axis, false);
  f_sum->setup(Variables{&g_pow}, Variables{&sum});
  f_sum->forward(Variables{&g_pow}, Variables{&sum});

  const T *l2sum = sum.get_data_pointer<T>(ctx);
  T *grad = param->cast_grad_and_get_pointer<T>(ctx);
  Size_t size = param->size();
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_clip_grad_by_norm, size, grad, l2sum,
                                 clip_norm);
}
} // namespace nbla
#endif
