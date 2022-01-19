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

#ifndef NBLA_CUDA_FUNCTION_STFT_HPP
#define NBLA_CUDA_FUNCTION_STFT_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/function/utils/stft.hpp>
#include <nbla/function/stft.hpp>

namespace nbla {

// Prototype declaration for cross referencing between STFTCuda and ISTFTCuda.
template <typename T> class ISTFTCuda;

template <typename T> class STFTCuda : public STFT<T> {
  // ISTFTCuda needs some STFTCuda methods when `as_stft_backward == true`.
  friend ISTFTCuda<T>;

  stft::WINDOW_TYPE window_type_t_;

protected:
  // Only for `as_istft_backward == true`.
  shared_ptr<ISTFTCuda<T>> istft_cuda_;

public:
  typedef typename CudaType<T>::type Tcu;

  explicit STFTCuda(const Context &ctx, int window_size, int stride,
                    int fft_size, const string &window_type, bool center,
                    const string &pad_mode, bool as_istft_backward)
      : STFT<T>(ctx, window_size, stride, fft_size, window_type, center,
                pad_mode, as_istft_backward),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~STFTCuda() {}
  virtual string name() { return "STFTCuda"; }
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
  virtual void calculate_conv_weight(Variable &conv_r, Variable &conv_i);

  // Only for `as_istft_backward == true`.
  virtual void apply_inv_window_forward(Variable *x, Variable *y);
  virtual void apply_inv_window_backward(Variable *x, Variable *y,
                                         const bool accum);
};
}
#endif
