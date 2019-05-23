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

/** AveragePooling
*/
#ifndef __NBLA_CUDA_CUDNN_FUNCTION_AVERAGEPOOLING_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_AVERAGEPOOLING_HPP__

#include <nbla/function/average_pooling.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/utils/base_pooling.hpp>

namespace nbla {

template <typename T>
class AveragePoolingCudaCudnn : public BasePoolingCudaCudnn<AveragePooling<T>> {
public:
  AveragePoolingCudaCudnn(const Context &ctx, const vector<int> &kernel,
                          const vector<int> &stride, bool ignore_border,
                          const vector<int> &pad, bool channel_last,
                          bool including_pad)
      : BasePoolingCudaCudnn<AveragePooling<T>>(ctx, kernel, stride,
                                                ignore_border, pad,
                                                channel_last, including_pad) {}
  string name() override { return "AveragePoolingCudaCudnn"; }
  cudnnPoolingMode_t mode() const override {
    return this->including_pad_ ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
                                : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
  }
};
}
#endif
