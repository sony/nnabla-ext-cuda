// Copyright 2021 Sony Corporation.
// Copyright 2021,2022 Sony Group Corporation.
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

#ifndef NBLA_CUDA_FUNCTION_ISTFT_HPP
#define NBLA_CUDA_FUNCTION_ISTFT_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/function/utils/stft.hpp>
#include <nbla/function/istft.hpp>

namespace nbla {

// Prototype declaration for cross referencing between STFTCuda and ISTFTCuda.
template <typename T> class STFTCuda;

template <typename T> class ISTFTCuda : public ISTFT<T> {
  // STFTCuda needs some ISTFTCuda methods when `as_istft_backward == true`.
  friend STFTCuda<T>;

  stft::WINDOW_TYPE window_type_t_;

protected:
  // Only for `as_istft_backward == true`.
  shared_ptr<STFTCuda<T>> stft_cuda_;

public:
  typedef typename CudaType<T>::type Tcu;

  explicit ISTFTCuda(const Context &ctx, int window_size, int stride,
                     int fft_size, const string &window_type, bool center,
                     const string &pad_mode, bool as_stft_backward)
      : ISTFT<T>(ctx, window_size, stride, fft_size, window_type, center,
                 pad_mode, as_stft_backward),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~ISTFTCuda() {}
  virtual string name() { return "ISTFTCuda"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int device_;
  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
  virtual void calculate_window(Context &ctx, Variable *window) const;
  virtual void calculate_inv_window(Context &ctx, Variable *inv_window);
  virtual void calculate_conv_weight(Variable &conv_cos, Variable &conv_sin);
  virtual void apply_inv_window_forward(Variable *x, Variable *y);
  virtual void apply_inv_window_backward(Variable *x, Variable *y,
                                         const bool accum);
};
}
#endif
