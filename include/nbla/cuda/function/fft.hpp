// Copyright (c) 2017-2020 Sony Corporation. All Rights Reserved.
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

#ifndef NBLA_CUDA_FUNCTION_FFT_HPP
#define NBLA_CUDA_FUNCTION_FFT_HPP

#include <cufft.h>
#include <cufftXt.h>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/function/utils/fft.hpp>
#include <nbla/function/fft.hpp>

namespace nbla {

template <typename T> class FFTCuda : public FFT<T> {

public:
  typedef typename CudaType<T>::type Tcu;

  explicit FFTCuda(const Context &ctx, int signal_ndim, bool normalized)
      : FFT<T>(ctx, signal_ndim, normalized), signal_size_(1),
        device_(std::stoi(ctx.device_id)) {
    cuda_set_device(this->device_);
    NBLA_CUFFT_CHECK(cufftCreate(&plan_forward_));
    NBLA_CUFFT_CHECK(cufftCreate(&plan_backward_));
  }
  virtual ~FFTCuda();
  virtual string name() { return "FFTCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  Size_t signal_size_;
  int device_;
  cufftHandle plan_forward_;
  cufftHandle plan_backward_;
  vector<long long int> n_;

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif
