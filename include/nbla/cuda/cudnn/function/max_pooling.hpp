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

/** MaxPooling
*/
#ifndef __NBLA_CUDA_CUDNN_FUNCTION_MAXPOOLING_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_MAXPOOLING_HPP__

#include <nbla/function/max_pooling.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/utils/base_pooling.hpp>

namespace nbla {

template <typename T>
class MaxPoolingCudaCudnn
    : public BasePoolingCudaCudnn<typename MaxPooling<T>::base_pooling_type> {
public:
  MaxPoolingCudaCudnn(const Context &ctx, const vector<int> &kernel,
                      const vector<int> &stride, bool ignore_border,
                      const vector<int> &pad, bool channel_last)
      : BasePoolingCudaCudnn<typename MaxPooling<T>::base_pooling_type>(
            ctx, kernel, stride, ignore_border, pad, channel_last) {}
  string name() override { return "MaxPoolingCudaCudnn"; }
  cudnnPoolingMode_t mode() const override {
    auto cudnn_handle_manager = SingletonManager::get<CudnnHandleManager>();
    bool deterministic = cudnn_handle_manager->get_deterministic_option();
    if (deterministic)
      return CUDNN_POOLING_MAX_DETERMINISTIC;
    return CUDNN_POOLING_MAX;
  }
};
}
#endif
