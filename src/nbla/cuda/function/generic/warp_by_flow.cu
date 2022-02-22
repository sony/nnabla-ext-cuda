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

#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/warp_by_flow.hpp>
#include <nbla/cuda/utils/atomic_add.cuh>
#include <nbla/variable.hpp>

namespace nbla {

namespace warp_by_flow {

template <typename T1, typename T2>
__device__ inline T2 clamp_to_index(const T1 value, const T2 maximum) {
  return max(T2(0), min(maximum, T2(value)));
}

template <typename T>
__global__ void forward(const int size, const int4 oshape, const int4 strides,
                        const T *data, const T *flow, T *out) {
  // size = NCHW, oshape.(w|z|y|x) = (N|C|H|W), stride.(w|z|y|x) = (CHW|HW|W|1)
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    auto x = index;
    auto n = x / strides.w;
    x -= n * strides.w;
    auto c = x / strides.z;
    x -= c * strides.z;
    auto y = x / strides.y;
    x -= y * strides.y;

    auto xf = T(x) + flow[(n * 2 + 0) * strides.z + y * strides.y + x];
    auto yf = T(y) + flow[(n * 2 + 1) * strides.z + y * strides.y + x];
    auto xl = clamp_to_index(floor(xf), oshape.x - 1);
    auto yt = clamp_to_index(floor(yf), oshape.y - 1);
    auto xr = clamp_to_index(floor(xf) + 1, oshape.x - 1);
    auto yb = clamp_to_index(floor(yf) + 1, oshape.y - 1);
    auto tl = data[n * strides.w + c * strides.z + yt * strides.y + xl];
    auto tr = data[n * strides.w + c * strides.z + yt * strides.y + xr];
    auto bl = data[n * strides.w + c * strides.z + yb * strides.y + xl];
    auto br = data[n * strides.w + c * strides.z + yb * strides.y + xr];
    auto a0 = xf - T(xl), b0 = yf - T(yt);
    auto a1 = T(1) - a0, b1 = T(1) - b0;
    out[index] = a1 * b1 * tl + a1 * b0 * bl + a0 * b1 * tr + a0 * b0 * br;
  }
}

template <typename T>
__global__ void grad2data(const int size, const int4 oshape, const int4 stride,
                          const T *data, const T *flow, const T *grad, T *out) {
  // size = NCHW, oshape.(w|z|y|x) = (N|C|H|W), stride.(w|z|y|x) = (CHW|HW|W|1)
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    auto x = index;
    auto n = x / stride.w;
    x -= n * stride.w;
    auto c = x / stride.z;
    x -= c * stride.z;
    auto y = x / stride.y;
    x -= y * stride.y;

    auto xf = T(x) + flow[(n * 2 + 0) * stride.z + y * stride.y + x];
    auto yf = T(y) + flow[(n * 2 + 1) * stride.z + y * stride.y + x];
    auto xl = clamp_to_index(floor(xf), oshape.x - 1);
    auto yt = clamp_to_index(floor(yf), oshape.y - 1);
    auto xr = clamp_to_index(floor(xf) + 1, oshape.x - 1);
    auto yb = clamp_to_index(floor(yf) + 1, oshape.y - 1);
    auto a0 = xf - T(xl), b0 = yf - T(yt);
    auto a1 = T(1) - a0, b1 = T(1) - b0;
    auto nc = n * stride.w + c * stride.z;
    atomic_add(&out[nc + yt * stride.y + xl], a1 * b1 * grad[index]);
    atomic_add(&out[nc + yb * stride.y + xl], a1 * b0 * grad[index]);
    atomic_add(&out[nc + yt * stride.y + xr], a0 * b1 * grad[index]);
    atomic_add(&out[nc + yb * stride.y + xr], a0 * b0 * grad[index]);
  }
}

template <typename T, bool accum>
__global__ void grad2flow(const int size, const int4 oshape, const int4 stride,
                          const T *data, const T *flow, const T *grad, T *out) {
  // size = N2HW, oshape.(w|z|y|x) = (N|C|H|W), stride.(w|z|y|x) = (CHW|HW|W|1)
  NBLA_CUDA_KERNEL_LOOP(index, size) {
    auto x = index;
    auto n = x / (2 * stride.z);
    x -= n * (2 * stride.z);
    auto c = x / stride.z;
    x -= c * stride.z;
    auto y = x / stride.y;
    x -= y * stride.y;

    auto xf = T(x) + flow[(n * 2 + 0) * stride.z + y * stride.y + x];
    auto yf = T(y) + flow[(n * 2 + 1) * stride.z + y * stride.y + x];
    auto xl = clamp_to_index(floor(xf), oshape.x - 1);
    auto yt = clamp_to_index(floor(yf), oshape.y - 1);
    auto xr = clamp_to_index(floor(xf) + 1, oshape.x - 1);
    auto yb = clamp_to_index(floor(yf) + 1, oshape.y - 1);
    auto gamma = c == 0 ? T(yb) - yf : T(xr) - xf;
    auto one_minus_gamma = T(1) - gamma;
    auto value = T(0);

    if (c == 0) {
      for (decltype(oshape.z) z = 0; z < oshape.z; z++) {
        auto tl = data[n * stride.w + z * stride.z + yt * stride.y + xl];
        auto tr = data[n * stride.w + z * stride.z + yt * stride.y + xr];
        auto bl = data[n * stride.w + z * stride.z + yb * stride.y + xl];
        auto br = data[n * stride.w + z * stride.z + yb * stride.y + xr];
        auto g = grad[n * stride.w + z * stride.z + y * stride.y + x];
        value += (gamma * (tr - tl) + one_minus_gamma * (br - bl)) * g;
      }
    } else {
      for (decltype(oshape.z) z = 0; z < oshape.z; z++) {
        auto tl = data[n * stride.w + z * stride.z + yt * stride.y + xl];
        auto tr = data[n * stride.w + z * stride.z + yt * stride.y + xr];
        auto bl = data[n * stride.w + z * stride.z + yb * stride.y + xl];
        auto br = data[n * stride.w + z * stride.z + yb * stride.y + xr];
        auto g = grad[n * stride.w + z * stride.z + y * stride.y + x];
        value += (gamma * (bl - tl) + one_minus_gamma * (br - tr)) * g;
      }
    }
    out[index] = accum ? out[index] + value : value;
  }
}
}

template <typename T>
void WarpByFlowCuda<T>::setup_impl(const Variables &inputs,
                                   const Variables &outputs) {
  WarpByFlow<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void WarpByFlowCuda<T>::forward_impl(const Variables &inputs,
                                     const Variables &outputs) {
  cuda_set_device(this->device_);

  auto data = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto flow = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  auto out = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  auto N = outputs[0]->shape().at(0);
  auto C = outputs[0]->shape().at(1);
  auto H = outputs[0]->shape().at(2);
  auto W = outputs[0]->shape().at(3);

  auto oshape = make_int4(W, H, C, N);
  auto stride = make_int4(1, W, H * W, C * H * W);

  using warp_by_flow::forward;
  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(forward, outputs[0]->size(), oshape, stride,
                                 data, flow, out);
}

template <typename T>
void WarpByFlowCuda<T>::backward_impl(const Variables &inputs,
                                      const Variables &outputs,
                                      const vector<bool> &propagate_down,
                                      const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1])) {
    return;
  }
  cuda_set_device(this->device_);

  auto grad = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  auto data = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  auto flow = inputs[1]->get_data_pointer<Tcu>(this->ctx_);

  auto N = outputs[0]->shape().at(0);
  auto C = outputs[0]->shape().at(1);
  auto H = outputs[0]->shape().at(2);
  auto W = outputs[0]->shape().at(3);

  auto oshape = make_int4(W, H, C, N);
  auto stride = make_int4(1, W, H * W, C * H * W);

  if (propagate_down[0]) {
    if (!accum[0])
      inputs[0]->grad()->zero();
    using warp_by_flow::grad2data;
    auto g_data = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(grad2data, inputs[0]->size(), oshape, stride,
                                   data, flow, grad, g_data);
  }

  if (propagate_down[1]) {
    using warp_by_flow::grad2flow;
    auto wronly = !accum[1]; // if we shall only write the gradient, not update
    auto g_flow = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, wronly);
    auto kernel = accum[1] ? (grad2flow<Tcu, true>) : (grad2flow<Tcu, false>);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, inputs[1]->size(), oshape, stride,
                                   data, flow, grad, g_flow);
  }
}
}
