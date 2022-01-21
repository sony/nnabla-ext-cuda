// Copyright 2020,2021 Sony Corporation.
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
#include <nbla/cuda/function/add_n.hpp>
#include <nbla/cuda/utils/pointers.cuh>
#include <nbla/variable.hpp>

using std::count;

namespace nbla {

template <typename T>
__global__ void kernel_add_n_forward(const int size, const int num_inputs,
                                     const T **x, T *y) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) {
    T val = 0;
    for (int i = 0; i < num_inputs; i++) {
      val += x[i][idx];
    }
    y[idx] = val;
  }
}

template <typename T>
__global__ void
kernel_add_n_backward(const int num, const int num_inputs, T **dx, const T *dy,
                      const uint8_t *propdown, const uint8_t *accum) {
  NBLA_CUDA_KERNEL_LOOP(idx, num) {
    for (int i = 0; i < num_inputs; i++) {
      if (propdown[i])
        dx[i][idx] = (accum[i] ? dx[i][idx] : (T)0) + dy[idx];
    }
  }
}

template <typename T>
__global__ void kernel_accum1_backward(const int size, const T *dy, T *dx) {
  NBLA_CUDA_KERNEL_LOOP(idx, size) { dx[idx] += dy[idx]; }
}

template <typename T>
void AddNCuda<T>::setup_impl(const Variables &inputs,
                             const Variables &outputs) {
  AddN<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);
}

template <typename T>
void AddNCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {
  cuda_set_device(this->device_);

  auto input_mask = this->cg_input_mask;
  auto n_active_inputs = std::count(input_mask.begin(), input_mask.end(), true);
  if (n_active_inputs == 1) {
    auto input_index =
        std::distance(input_mask.begin(),
                      std::find(input_mask.begin(), input_mask.end(), true));
    auto x = inputs.at(input_index)->data()->get(get_dtype<Tcu>(), this->ctx_);
    auto y = outputs.at(0)->data()->cast(get_dtype<Tcu>(), this->ctx_, true);
    y->copy_from(x);
  } else if (n_active_inputs > 1) {
    Variables _inputs;
    for (size_t i = 0; i < inputs.size(); i++) {
      if (input_mask.at(i) == true) {
        _inputs.push_back(inputs[i]);
      }
    }
    auto y = outputs.at(0)->cast_data_and_get_pointer<Tcu>(this->ctx_, true);
    auto xptrs = get_cuda_pointer_array<Tcu>(_inputs, this->ctx_, [&](int i) {
      return _inputs[i]->template get_data_pointer<Tcu>(this->ctx_);
    });
    auto kernel = kernel_add_n_forward<Tcu>;
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, outputs[0]->size(), _inputs.size(),
                                   xptrs->template pointer<const Tcu *>(), y);
  }
}

template <typename T>
void AddNCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {
  cuda_set_device(std::stoi(this->ctx_.device_id));

  auto input_mask = this->cg_input_mask;
  auto n_active_inputs = std::count(input_mask.begin(), input_mask.end(), true);
  if (n_active_inputs == 1) {
    auto input_index =
        std::distance(input_mask.begin(),
                      std::find(input_mask.begin(), input_mask.end(), true));
    if (propagate_down.at(input_index)) {
      Variable *input = inputs.at(input_index);
      if (accum.at(input_index)) {
        auto dx = input->cast_grad_and_get_pointer<Tcu>(this->ctx_, false);
        auto dy = outputs.at(0)->get_data_pointer<Tcu>(this->ctx_);
        auto kernel = kernel_accum1_backward<Tcu>;
        NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel, outputs.at(0)->size(), dy, dx);
      } else {
        auto dx = input->grad()->cast(get_dtype<Tcu>(), this->ctx_, true);
        auto dy = outputs.at(0)->grad()->get(get_dtype<Tcu>(), this->ctx_);
        dx->copy_from(dy);
      }
    }
  } else if (n_active_inputs > 1) {
    auto dy = outputs.at(0)->get_grad_pointer<Tcu>(this->ctx_);
    Variables _inputs;
    vector<bool> _propagate_down, _accum;
    for (size_t i = 0; i < inputs.size(); i++) {
      if (input_mask.at(i) == true) {
        _inputs.push_back(inputs[i]);
        _propagate_down.push_back(propagate_down.at(i));
        _accum.push_back(accum.at(i));
      }
    }
    auto dxptrs = get_cuda_pointer_array<Tcu>(_inputs, this->ctx_, [&](int i) {
      return _inputs.at(i)->template cast_grad_and_get_pointer<Tcu>(this->ctx_,
                                                                    !_accum[i]);
    });
    auto propdown_array =
        create_ndarray_from_vector<bool, uint8_t>(_propagate_down);
    auto accum_array = create_ndarray_from_vector<bool, uint8_t>(_accum);
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
        (kernel_add_n_backward<Tcu>), outputs[0]->size(), _inputs.size(),
        dxptrs->template pointer<Tcu *>(), dy,
        propdown_array->cast(get_dtype<uint8_t>(), this->ctx_)
            ->template const_pointer<uint8_t>(),
        accum_array->cast(get_dtype<uint8_t>(), this->ctx_)
            ->template const_pointer<uint8_t>());
  }
}
}
