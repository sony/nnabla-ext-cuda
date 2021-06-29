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

  BatchNormalizationCuda<T> batch_norm_;

  // Temporally buffers and functions for forward
  Variable v_staging_data_for_forward_;
  Variable v_semaphores_for_forward_;
  Variable v_local_mean_, v_local_invstd_, v_local_count_;
  Variable v_all_mean_, v_all_invstd_, v_all_count_;
  Variable v_all_gather_send_;
  Variable v_all_gather_recv_;

  FunctionPtr concatenate_;
  FunctionPtr slice_mean_;
  FunctionPtr slice_invstd_;
  FunctionPtr slice_count_;

  // Temporally buffers and functions for backward
  Variable v_staging_data_for_backward_;
  Variable v_semaphores_for_backward_;
  Variable v_sum_dy_o_, v_sum_dy_xmu_o_;
  Variable v_beta_grad_, v_gamma_grad_;

  FunctionPtr add2_;

public:
  typedef typename CudaType<T>::type Tc;

  SyncBatchNormalizationCuda(const Context &ctx,
                             const std::shared_ptr<Communicator> &comm,
                             const std::string &group, const vector<int> axes,
                             float decay_rate, float eps, bool batch_stat)
      : SyncBatchNormalization<T>(ctx, comm, group, axes, decay_rate, eps,
                                  batch_stat),
        device_(std::stoi(ctx.device_id)),
        batch_norm_(ctx, axes, decay_rate, eps, batch_stat,
                    false /* no_scale */, false /* no_scale */) {}
  virtual ~SyncBatchNormalizationCuda() {}
  virtual string name() override { return "SyncBatchNormalizationCuda"; }
  virtual vector<string> allowed_array_classes() override {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl_batch(const Variables &inputs,
                                  const Variables &outputs,
                                  const bool update_inputs) override;
  virtual void forward_impl_global(const Variables &inputs,
                                   const Variables &outputs) override;
  virtual void backward_impl_batch(const Variables &inputs,
                                   const Variables &outputs,
                                   const vector<bool> &propagate_down,
                                   const vector<bool> &accum) override;
};
}
#endif
