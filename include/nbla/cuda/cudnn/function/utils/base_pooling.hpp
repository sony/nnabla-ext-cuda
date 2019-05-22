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
#pragma once

#include <nbla/function/utils/base_pooling.hpp>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

namespace nbla {

/** Base class of CUDNN Pooling functions.
 */
template <typename BasePoolingType>
class BasePoolingCudaCudnn : public BasePoolingType {
  int device_;
  CudnnPooling::Ptr cudnn_pooling_;

public:
  typedef BasePoolingCudaCudnn<typename BasePoolingType::base_pooling_type>
      base_pooling_cuda_cudnn_type;
  typedef typename BasePoolingType::data_type T;
  typedef typename CudaType<T>::type Tw;

  template <typename... Args>
  BasePoolingCudaCudnn<BasePoolingType>(const Context &ctx, Args... args)
      : BasePoolingType(ctx, args...), device_(std::stoi(ctx.device_id)) {}
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  void setup_impl(const Variables &inputs, const Variables &outputs) override;
  void forward_impl(const Variables &inputs, const Variables &outputs) override;
  void backward_impl(const Variables &inputs, const Variables &outputs,
                     const vector<bool> &propagate_down,
                     const vector<bool> &accum) override;
  /* Override this to determine cudnnPoolingMode_t fed to API.
   */
  virtual cudnnPoolingMode_t mode() const = 0;
}; // class BasePoolingCudaCudnn

} // namespace nbla
