// Copyright 2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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

/** Batch Normalization
 */
#ifndef __NBLA_CUDA_CUDNN_FUNCTION_SYNC_BATCHNORM_HPP__
#define __NBLA_CUDA_CUDNN_FUNCTION_SYNC_BATCHNORM_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/batch_normalization.hpp>
#include <nbla/cuda/function/sync_batch_normalization.hpp>

#include <vector>

using std::vector;

namespace nbla {

template <typename T>
class SyncBatchNormalizationCudaCudnn : public SyncBatchNormalizationCuda<T> {
protected:
  BatchNormalizationCudaCudnn<T> batch_norm_cudnn_;

public:
  typedef typename CudaType<T>::type Tw;

  SyncBatchNormalizationCudaCudnn(const Context &ctx,
                                  const std::shared_ptr<Communicator> &comm,
                                  const std::string &group,
                                  const vector<int> axes, float decay_rate,
                                  float eps, bool batch_stat)
      : SyncBatchNormalizationCuda<T>(ctx, comm, group, axes, decay_rate, eps,
                                      batch_stat),
        batch_norm_cudnn_(ctx, axes, decay_rate, eps, batch_stat,
                          false /* no_scale */, false /* no_bias */) {}
  virtual string name() override { return "SyncBatchNormalizationCudaCudnn"; }
  virtual vector<string> allowed_array_classes() override {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl_global(const Variables &inputs,
                                   const Variables &outputs) override;
};
} // namespace nbla
#endif
