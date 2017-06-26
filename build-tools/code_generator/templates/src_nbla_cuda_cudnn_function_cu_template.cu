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

// -*- coding:utf-8 -*-
/*
 * Copyright (C) 2016 Sony Corporation
 * This is UNPUBLISHED PROPRIETARY SOURCE CODE of Sony Corporation;
 * the contents of this file is not to be disclosed to third parties, copied
 * or duplicated in any form, in whole or in part, without the prior written
 * permission of Sony Corporation.
 *
 * *WARNING*
 * THIS FILE IS AUTO-GENERATED DUMMY CODE BY CODE GENERATOR.
 * PLEASE IMPLEMENT REAL CODE AND DELETE THIS MESSAGE SOON.
 * If you want to change dummy code, edit following files.
 * - build-tools/code_generator/function_generator/generate_src_nbla_cuda_cudnn_function_cu.py
 * - build-tools/code_generator/templates/src_nbla_cuda_cudnn_function_cu_template.cu
 */

/** {func_name}
 */

#include <algorithm>
#include <nbla/array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/{func_name_snakecase}.hpp>
#include <nbla/variable.hpp>

namespace nbla {{

template <{template_defines}>
void {func_name}CudaCudnn<{templates}>::setup_impl(const Variables &inputs,
                                     const Variables &outputs) {{
  // TODO TEMPLATE CODE
}}

template <{template_defines}>
void {func_name}CudaCudnn<{templates}>::forward_impl(const Variables &inputs,
                                       const Variables &outputs) {{
  // TODO TEMPLATE CODE
}}

template <{template_defines}>
void {func_name}CudaCudnn<{templates}>::backward_impl(const Variables &inputs,
                                        const Variables &outputs,
                                        const vector<bool> &propagate_down) {{
  // TODO TEMPLATE CODE
}}

template class {func_name}CudaCudnn<{ctypes}>;
}}

