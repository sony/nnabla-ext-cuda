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

/** Batch Normalization
 */
#ifndef __NBLA_CUDA_FUNCTION_SYNC_BATCHNORM_HPP__
#define __NBLA_CUDA_FUNCTION_SYNC_BATCHNORM_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/function/batch_normalization.hpp>
#include <nbla/function/sync_batch_normalization.hpp>

#include <vector>

using std::vector;

namespace nbla {

template <typename T>
class SyncBatchNormalizationCuda : public SyncBatchNormalization<T> {
protected:
  int device_;
  int blocks;
  // for transpose
  Variable v_axes_;
  Variable v_in_strides_;
  Variable v_out_strides_;
  Variable v_out_shape_;
  Variable v_in_shape_;
  Variable v_in_trans_;
  Variable v_din_trans_;
  // work memory for data of each axis
  Variable v_sync_;
  Variable v_dmean_;
  Variable v_dvar_;
  Variable v_t_;
  Variable v_inv_sqrt_variance_;
  // work memory for each block data of shuffle reduction
  Variable v_mean_reduction_space_;
  Variable v_variance_reduction_space_;
  Variable v_tmp_reduction_space_;

  BatchNormalizationCuda<T> batch_norm_;

public:
  typedef typename CudaType<T>::type Tc;

  SyncBatchNormalizationCuda(const Context &ctx,
                             const std::shared_ptr<Communicator> &comm,
                             const std::string &group, const vector<int> axes,
                             float decay_rate, float eps, bool batch_stat)
      : SyncBatchNormalization<T>(ctx, comm, group, axes, decay_rate, eps,
                                  batch_stat),
        device_(std::stoi(ctx.device_id)),
        batch_norm_(ctx, axes, decay_rate, eps, batch_stat) {}
  virtual ~SyncBatchNormalizationCuda() {}
  virtual string name() override { return "SyncBatchNormalizationCuda"; }
  virtual vector<string> allowed_array_classes() override {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl_batch(const Variables &inputs,
                                  const Variables &outputs) override;
  virtual void forward_impl_global(const Variables &inputs,
                                   const Variables &outputs) override;
  virtual void backward_impl_batch(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) override;
};
}
#endif
