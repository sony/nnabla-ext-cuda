// Copyright 2018,2019,2020,2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

#ifndef NBLA_CUDA_CUDNN_FUNCTION_SUM_HPP
#define NBLA_CUDA_CUDNN_FUNCTION_SUM_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/function/sum.hpp>

namespace nbla {

template <typename T> class SumCudaCudnn : public SumCuda<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit SumCudaCudnn(const Context &ctx, const vector<int> &axes,
                        bool keep_dims)
      : SumCuda<T>(ctx, axes, keep_dims) {
#if CUDNN_VERSION >= 6000
    NBLA_CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&x_desc_));
    NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&y_desc_));
#else
    this->fall_back_func_ = make_shared<SumCuda<T>>(ctx, axes, keep_dims);
#endif
  }
  virtual ~SumCudaCudnn() {
#if CUDNN_VERSION >= 6000
    NBLA_CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reduce_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(x_desc_));
    NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(y_desc_));
#endif
  }
  virtual string name() { return "SumCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

#if CUDNN_VERSION >= 6000
protected:
  cudnnReduceTensorDescriptor_t reduce_desc_;
  cudnnTensorDescriptor_t x_desc_;
  cudnnTensorDescriptor_t y_desc_;
  size_t workspace_size_;
  bool same_in_out_shape_;

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
#endif
};
}
#endif
