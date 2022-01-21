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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/sync_batch_normalization.hpp>

namespace nbla {

template <typename T>
void SyncBatchNormalizationCudaCudnn<T>::setup_impl(const Variables &inputs,
                                                    const Variables &outputs) {
  this->batch_norm_cudnn_.setup(inputs, outputs);

  SyncBatchNormalizationCuda<T>::setup_impl(inputs, outputs);
}

template <class T>
void SyncBatchNormalizationCudaCudnn<T>::forward_impl_global(
    const Variables &inputs, const Variables &outputs) {
  this->batch_norm_cudnn_.forward(inputs, outputs);
}
}