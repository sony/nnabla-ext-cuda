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
#include <nbla/cuda/function/pack_padded_sequence.hpp>
#include <nbla/cuda/function/utils/rnn.cuh>
#include <nbla/function/transpose.hpp>
#include <nbla/function/utils/rnn.hpp>
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <numeric>

namespace nbla {

template <typename U>
void PackPaddedSequenceCuda<U>::setup_impl(const Variables &inputs,
                                           const Variables &outputs) {
  PackPaddedSequence<U>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  if (this->batch_first_) {
    auto padded_sequence = inputs[0];
    vector<int> axes(padded_sequence->ndim());
    std::iota(axes.begin(), axes.end(), 0);
    std::iter_swap(axes.begin(), axes.begin() + 1);
    f_transpose_ = create_Transpose(this->ctx_, axes);
  }
}

template <typename U>
void PackPaddedSequenceCuda<U>::forward_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(this->device_);
  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto padded_sequence = inputs[0];
  auto lengths = inputs[1];
  auto packed_sequence = outputs[0];
  auto batch_sizes = outputs[1];

  auto N = packed_sequence->shape()[0];
  auto T = batch_sizes->shape()[0];
  auto B = lengths->shape()[0];
  auto D = packed_sequence->ndim() == 1 ? 1 : packed_sequence->size(1);
  auto ishape0 = padded_sequence->shape();
  auto oshape0 = packed_sequence->shape();

  // Transpose
  Variable padded_sequence_t;
  if (this->batch_first_)
    execute(f_transpose_, {padded_sequence}, {&padded_sequence_t});
  auto _get_data = [this](Variable *v) {
    return v->get_data_pointer<Tcu>(this->ctx_);
  };

  // Pack
  auto data_padded_sequence = this->batch_first_ ? _get_data(&padded_sequence_t)
                                                 : _get_data(padded_sequence);
  auto data_lengths = lengths->get_data_pointer<int>(cpu_ctx);
  auto data_packed_sequence =
      packed_sequence->cast_data_and_get_pointer<Tcu>(this->ctx_);
  auto data_batch_sizes = batch_sizes->cast_data_and_get_pointer<int>(cpu_ctx);

  using function::utils::rnn::compute_batch_sizes;
  using cuda::function::utils::rnn::pack;
  compute_batch_sizes(data_lengths, lengths->size(), data_batch_sizes);
  pack<Tcu, false>(this->ctx_, data_padded_sequence, data_batch_sizes,
                   data_packed_sequence, T, B, D, N);
}

template <typename U>
void PackPaddedSequenceCuda<U>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  cuda_set_device(this->device_);
  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto padded_sequence = inputs[0];
  auto lengths = inputs[1];
  auto packed_sequence = outputs[0];
  auto batch_sizes = outputs[1];

  auto N = packed_sequence->shape()[0];
  auto T = batch_sizes->shape()[0];
  auto B = lengths->shape()[0];
  auto D = packed_sequence->ndim() == 1 ? 1 : packed_sequence->size(1);
  auto ishape0 = padded_sequence->shape();
  auto oshape0 = packed_sequence->shape();

  // Unpack
  Variable padded_sequence_t(Shape_t{T, B, D});
  auto _cast_grad = [this](Variable *v) {
    return v->cast_grad_and_get_pointer<Tcu>(this->ctx_);
  };
  auto grad_padded_sequence = this->batch_first_
                                  ? _cast_grad(&padded_sequence_t)
                                  : _cast_grad(padded_sequence);
  auto grad_packed_sequence =
      packed_sequence->get_grad_pointer<Tcu>(this->ctx_);
  auto data_batch_sizes = batch_sizes->cast_data_and_get_pointer<int>(cpu_ctx);

  using cuda::function::utils::rnn::unpack;
  if (accum[0] && !this->batch_first_) {
    unpack<Tcu, true>(this->ctx_, grad_packed_sequence, data_batch_sizes,
                      grad_padded_sequence, T, B, D, N);
  } else {
    unpack<Tcu, false>(this->ctx_, grad_packed_sequence, data_batch_sizes,
                       grad_padded_sequence, T, B, D, N);
  }

  // Tranpose
  if (this->batch_first_)
    nbla::backward(f_transpose_, {padded_sequence}, {&padded_sequence_t},
                   {true}, {accum[0]}, true);
}
}
