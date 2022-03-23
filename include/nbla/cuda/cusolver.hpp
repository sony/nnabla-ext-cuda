// Copyright 2022 Sony Group Corporation.
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

#ifndef NBLA_CUSOLVER_HPP
#define NBLA_CUSOLVER_HPP

#include <cusolverDn.h>

namespace nbla {
template <typename T>
void cusolverdn_potrf_batched(cusolverDnHandle_t handle, int n, T **x, int lda,
                              int *info, int batchSize);
}
#endif
