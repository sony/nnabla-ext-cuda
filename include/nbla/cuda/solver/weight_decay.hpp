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

#ifndef __NBLA_CUDA_SOLVER_WEIGHT_DECAY_HPP__
#define __NBLA_CUDA_SOLVER_WEIGHT_DECAY_HPP__

#include <nbla/solver/sgd.hpp>

namespace nbla {

template <typename T>
void weight_decay_cuda(const Context &ctx, const shared_ptr<Variable> param,
                       float decay_rate);

#define NBLA_DECL_WEIGHT_DECAY()                                               \
  virtual void weight_decay_impl(const string &key, VariablePtr param,         \
                                 float decay_rate)

#define NBLA_DEF_WEIGHT_DECAY(SOLVER, WEIGHT_DECAY_FUNC)                       \
  template <typename T>                                                        \
  void SOLVER<T>::weight_decay_impl(const string &key, VariablePtr param,      \
                                    float decay_rate) {                        \
    WEIGHT_DECAY_FUNC<T>(this->ctx_, param, decay_rate);                       \
  }
}
#endif
