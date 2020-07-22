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

#ifndef __NBLA_CUDNN_INIT_HPP__
#define __NBLA_CUDNN_INIT_HPP__
#include <nbla/cuda/defs.hpp>

namespace nbla {
/**
Initialize CUDNN features.
*/
NBLA_CUDA_API void init_cudnn();

/**
Set conv algo to blacklist.
*/
NBLA_CUDA_API void set_conv_fwd_algo_blacklist(int id);
NBLA_CUDA_API void set_conv_bwd_data_algo_blacklist(int id);
NBLA_CUDA_API void set_conv_bwd_filter_algo_blacklist(int id);

/**
Unset conv algo to blacklist.
*/
NBLA_CUDA_API void unset_conv_fwd_algo_blacklist(int id);
NBLA_CUDA_API void unset_conv_bwd_data_algo_blacklist(int id);
NBLA_CUDA_API void unset_conv_bwd_filter_algo_blacklist(int id);

}

#endif
