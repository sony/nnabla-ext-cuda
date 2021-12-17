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
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/pad_packed_sequence.hpp>
#include <nbla/cuda/function/utils/rnn.cuh>
#include <nbla/function/transpose.hpp>
#include <nbla/function/utils/rnn.hpp>
#include <nbla/imperative.hpp>
#include <nbla/variable.hpp>

#include <algorithm>
#include <numeric>

namespace nbla {

template <typename U>
void PadPackedSequenceCuda<U>::setup_impl(const Variables &inputs,
                                          const Variables &outputs) {
  PadPackedSequence<U>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

  if (this->batch_first_) {
    vector<int> axes(3);
    std::iota(axes.begin(), axes.end(), 0);
    std::iter_swap(axes.begin(), axes.begin() + 1);
    f_transpose_ = create_Transpose(this->ctx_, axes);
  }
}

template <typename U>
void PadPackedSequenceCuda<U>::forward_impl(const Variables &inputs,
                                            const Variables &outputs) {
  cuda_set_device(this->device_);
  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto packed_sequence = inputs[0];
  auto batch_sizes = inputs[1];
  auto padded_sequence = outputs[0];
  auto lengths = outputs[1];

  auto N = packed_sequence->shape()[0];
  auto T = batch_sizes->shape()[0];
  auto B = lengths->shape()[0];
  auto D = packed_sequence->ndim() == 1 ? 1 : packed_sequence->size(1);
  auto ishape0 = packed_sequence->shape();
  auto oshape0 = padded_sequence->shape();

  // Unpack
  auto TL = this->total_length_;
  Variable padded_sequence_t(Shape_t{(TL > T) ? TL : T, B, D});
  auto _cast_data = [this](Variable *v) {
    return v->cast_data_and_get_pointer<Tcu>(this->ctx_);
  };
  auto data_packed_sequence =
      packed_sequence->get_data_pointer<Tcu>(this->ctx_);
  auto data_batch_sizes = batch_sizes->get_data_pointer<int>(cpu_ctx);
  auto data_padded_sequence = this->batch_first_
                                  ? _cast_data(&padded_sequence_t)
                                  : _cast_data(padded_sequence);
  auto data_lengths = lengths->cast_data_and_get_pointer<int>(cpu_ctx);

  using utils::rnn::compute_lengths;
  using cuda::function::utils::rnn::unpack;
  compute_lengths(data_batch_sizes, batch_sizes->size(), data_lengths);
  unpack<Tcu, false>(this->ctx_, data_packed_sequence, data_batch_sizes,
                     data_padded_sequence, T, B, D, N, TL);
  // Transpose
  if (this->batch_first_) {
    auto oshape = padded_sequence->shape();
    execute(f_transpose_, {&padded_sequence_t}, {padded_sequence});
    padded_sequence->reshape(oshape, false);
  }
}

template <typename U>
void PadPackedSequenceCuda<U>::backward_impl(const Variables &inputs,
                                             const Variables &outputs,
                                             const vector<bool> &propagate_down,
                                             const vector<bool> &accum) {
  if (!propagate_down[0]) {
    return;
  }

  cuda_set_device(this->device_);
  nbla::Context cpu_ctx{{"cpu:int"}, "CpuCachedArray", "0"};
  auto packed_sequence = inputs[0];
  auto batch_sizes = inputs[1];
  auto padded_sequence = outputs[0];
  auto lengths = outputs[1];

  auto N = packed_sequence->shape()[0];
  auto T = batch_sizes->shape()[0];
  auto B = lengths->shape()[0];
  auto D = packed_sequence->ndim() == 1 ? 1 : packed_sequence->size(1);
  auto ishape0 = packed_sequence->shape();
  auto oshape0 = padded_sequence->shape();

  // Transpose
  auto TL = this->total_length_;
  Variable padded_sequence_t(Shape_t{(TL > T) ? TL : T, B, D});
  if (this->batch_first_) {
    auto oshape = padded_sequence->shape();
    nbla::backward(f_transpose_, {&padded_sequence_t}, {padded_sequence},
                   {true}, {false}, true);
    padded_sequence->reshape(oshape, false);
  }

  // Pack
  auto _get_grad = [this](Variable *v) {
    return v->get_grad_pointer<Tcu>(this->ctx_);
  };
  auto grad_packed_sequence =
      packed_sequence->cast_grad_and_get_pointer<Tcu>(this->ctx_);
  auto data_batch_sizes = batch_sizes->get_data_pointer<int>(cpu_ctx);
  auto grad_padded_sequence = this->batch_first_ ? _get_grad(&padded_sequence_t)
                                                 : _get_grad(padded_sequence);

  using cuda::function::utils::rnn::pack;
  if (accum[0]) {
    pack<Tcu, true>(this->ctx_, grad_padded_sequence, data_batch_sizes,
                    grad_packed_sequence, T, B, D, N);
  } else {
    pack<Tcu, false>(this->ctx_, grad_padded_sequence, data_batch_sizes,
                     grad_packed_sequence, T, B, D, N);
  }
}
}
