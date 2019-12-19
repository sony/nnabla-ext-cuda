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

#ifndef NBLA_CUDA_FUNCTION_BATCH_DET_HPP
#define NBLA_CUDA_FUNCTION_BATCH_DET_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/batch_det.hpp>

namespace nbla {

template <typename T> class BatchDetCuda : public BatchDet<T> {
public:
  /* TODO: remove this help message.
  Typedef of CUDA scalar types used in source file.
  This template function class might be instantiated for each CPU scalar types
  (double, float, nbla::Half), however, for Half, CUDA kernel functions
  must use nbla::HalfCuda in which a bunch of device operator functions are
  overloaded. nbla::CudaType<T>::type will translate nbla::Half
  to nbla::HalfCuda. For other types, it will keep it as-is.
  See nbla/cuda/half.hpp for other template utilities.
  */
  typedef typename CudaType<T>::type Tc;

  explicit BatchDetCuda(const Context &ctx)
      : BatchDet<T>(ctx), device_(std::stoi(ctx.device_id)) {}
  virtual ~BatchDetCuda() {}
  virtual string name() { return "BatchDetCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }
  virtual bool grad_depends_output_data(int i, int o) const { return true; }

protected:
  int device_;
  int dim_, batch_size_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif
