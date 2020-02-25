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

#ifndef __NBLA_CUDA_DEFS_HPP__
#define __NBLA_CUDA_DEFS_HPP__
// For windows support
#if defined(_MSC_VER) && !defined(__CUDACC__)
#if defined(nnabla_cuda_EXPORTS) || defined(nnabla_cuda_dbg_EXPORTS) ||        \
    defined(nnabla_cuda_80_7_EXPORTS) ||                                       \
    defined(nnabla_cuda_80_7_dbg_EXPORTS) ||                                   \
    defined(nnabla_cuda_90_7_EXPORTS) ||                                       \
    defined(nnabla_cuda_90_7_dbg_EXPORTS) ||                                   \
    defined(nnabla_cuda_100_7_EXPORTS) ||                                      \
    defined(nnabla_cuda_100_7_dbg_EXPORTS) ||                                  \
    defined(nnabla_cuda_101_7_EXPORTS) ||                                      \
    defined(nnabla_cuda_101_7_dbg_EXPORTS) ||                                  \
    defined(nnabla_cuda_102_7_EXPORTS) ||                                      \
    defined(nnabla_cuda_102_7_dbg_EXPORTS)
#define NBLA_CUDA_API __declspec(dllexport)
#else
#define NBLA_CUDA_API __declspec(dllimport)
#endif
#else
#define NBLA_CUDA_API
#endif
#endif
