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

/*
 * *WARNING*
 * THIS FILE IS AUTO-GENERATED DUMMY CODE BY CODE GENERATOR.
 * PLEASE IMPLEMENT REAL CODE AND DELETE THIS MESSAGE SOON.
 * If you want to change dummy code, edit following files.
 * - build-tools/code_generator/function_generator/generate_include_nbla_cuda_function_hpp.py
 * - build-tools/code_generator/templates/include_nbla_cuda_function_hpp_template.hpp
 */

/** {func_name}
 */
#ifndef __NBLA_CUDA_FUNCTION_{func_name_upcase}_HPP__
#define __NBLA_CUDA_FUNCTION_{func_name_upcase}_HPP__

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/function/{func_name_snakecase}.hpp>
namespace nbla {{
/** @copydoc {func_name}
*/

template <{template_defines}> class {func_name}Cuda : public {func_name}<{templates}> {{
public:
  explicit {func_name}Cuda({func_args})
    : {func_name}<{templates}>({func_arg_variables}), device_(std::stoi(ctx.device_id)) {{}}
  virtual ~{func_name}Cuda() {{}}
  virtual string name() {{ return "{func_name}Cuda"; }}
  virtual vector<string> allowed_array_classes() {{
    return SingletonManager::get<Cuda>()->array_classes();
  }}

protected:
  int device_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
}};
}}

#endif

