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

/** Communicator interface class
 */
#ifndef __NBLA_NCCL_NCCLUTILS_HPP__
#define __NBLA_NCCL_NCCLUTILS_HPP__
#include <nbla/array.hpp>
#include <nbla/context.hpp>
#include <nbla/variable.hpp>

#include <memory>
#include <string>
#include <unordered_map>

#include "nccl.h"

namespace nbla {

using std::string;
using std::vector;
using std::shared_ptr;
using std::unordered_map;
using std::pair;

#define NBLA_NCCL_CHECK(EXPRESSION)                                            \
  do {                                                                         \
    ncclResult_t ret = EXPRESSION;                                             \
    if (ret != ncclSuccess) {                                                  \
      NBLA_ERROR(error_code::target_specific,                                  \
                 "`" #EXPRESSION "` failed with %s.",                          \
                 ncclGetErrorString(ret));                                     \
    }                                                                          \
  } while (0)

template <typename Tc> ncclDataType_t get_nccl_dtype();

template <> inline ncclDataType_t get_nccl_dtype<float>() { return ncclFloat; }
template <> inline ncclDataType_t get_nccl_dtype<Half>() { return ncclHalf; }
template <> inline ncclDataType_t get_nccl_dtype<HalfCuda>() {
  return ncclHalf;
}
}
#endif
