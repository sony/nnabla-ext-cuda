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

#ifndef NBLA_CUDA_CUDNN_FUNCTION_RNN_HPP
#define NBLA_CUDA_CUDNN_FUNCTION_RNN_HPP

#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

#include <nbla/function/rnn.hpp>

namespace nbla {

struct WCudnnTensorDescArray {
private:
  bool initialized_;
  vector<cudnnTensorDescriptor_t> desc_array_;

public:
  vector<cudnnTensorDescriptor_t> &desc_array() { return desc_array_; }

  cudnnTensorDescriptor_t *data() {
    if (!initialized_) {
      return nullptr;
    }
    return desc_array_.data();
  }

  WCudnnTensorDescArray() : initialized_(false) {}

  WCudnnTensorDescArray(size_t size) : initialized_(true), desc_array_(size) {
    for (auto &desc : desc_array_) {
      NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    }
  }

  ~WCudnnTensorDescArray() {
    if (initialized_) {
      for (auto &desc : desc_array_) {
        NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
      }
    }
  }
  DISABLE_COPY_AND_ASSIGN(WCudnnTensorDescArray);
};
struct WCudnnTensorDesc {
  cudnnTensorDescriptor_t desc;
  WCudnnTensorDesc() { NBLA_CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc)); }
  ~WCudnnTensorDesc() { NBLA_CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc)); }
};

struct WCudnnFilterDesc {
  cudnnFilterDescriptor_t desc;
  WCudnnFilterDesc() { NBLA_CUDNN_CHECK(cudnnCreateFilterDescriptor(&desc)); }
  ~WCudnnFilterDesc() { NBLA_CUDNN_CHECK(cudnnDestroyFilterDescriptor(desc)); }
};

struct WCudnnDropoutDesc {
  cudnnDropoutDescriptor_t desc;
  WCudnnDropoutDesc() { NBLA_CUDNN_CHECK(cudnnCreateDropoutDescriptor(&desc)); }
  ~WCudnnDropoutDesc() {
    NBLA_CUDNN_CHECK(cudnnDestroyDropoutDescriptor(desc));
  }
};

struct WCudnnRNNDesc {
  cudnnRNNDescriptor_t desc;
  WCudnnRNNDesc() { NBLA_CUDNN_CHECK(cudnnCreateRNNDescriptor(&desc)); }
  ~WCudnnRNNDesc() { NBLA_CUDNN_CHECK(cudnnDestroyRNNDescriptor(desc)); }
};

template <typename T> class RNNCudaCudnn : public RNN<T> {
public:
  typedef typename CudaType<T>::type Tcu;

  explicit RNNCudaCudnn(const Context &ctx, int num_layers,
                        const string &nonlinearity, float dropout,
                        bool bidirectional, bool training)
      : RNN<T>(ctx, num_layers, nonlinearity, dropout, bidirectional, training),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~RNNCudaCudnn() {}
  virtual string name() { return "RNNCudaCudnn"; }
  virtual vector<string> allowed_array_classes() {
    return SingletonManager::get<Cuda>()->array_classes();
  }

protected:
  int seq_len_;
  int input_dim_;
  int hidden_size_;
  int num_directions_;
  int num_lin_layers_;
  bool weight_exists_;
  bool bias_exists_;

  size_t params_size_in_bytes_;
  size_t total_params_;

  int device_;
  std::unique_ptr<WCudnnTensorDescArray> x_desc_;

  WCudnnTensorDesc h_desc_;
  WCudnnFilterDesc params_desc_;
  std::unique_ptr<WCudnnTensorDescArray> y_desc_;
  WCudnnTensorDesc h_n_desc_;

  WCudnnTensorDesc c_x_desc_; // dummy
  WCudnnTensorDesc c_y_desc_; // dummy

  shared_ptr<CudaCachedArray> state_array_;
  WCudnnDropoutDesc dropout_desc_;
  WCudnnRNNDesc rnn_desc_;
  cudnnRNNInputMode_t inputMode;
  cudnnDirectionMode_t direction;
  cudnnRNNMode_t RNNMode;

  size_t workspace_size_;
  size_t reserve_size_{NULL};

  vector<pair<int, int>> weight_offsets_;
  vector<pair<int, int>> bias_offsets_;
  shared_ptr<CudaCachedArray> mem_reservespace_{nullptr};

  virtual void copy_weight_bias_to_params(Tcu *params, const Tcu *w_init,
                                          const Tcu *weight, const Tcu *bias,
                                          bool weight_exists, bool bias_exists);
  virtual void copy_params_to_gradients(Tcu *params, Tcu *w_init, Tcu *weight,
                                        Tcu *bias, bool w_init_accum,
                                        bool w_accum, bool b_accum,
                                        bool w_init_propagate, bool w_propagate,
                                        bool b_propagate);

  virtual void setup_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl(const Variables &inputs, const Variables &outputs);
  virtual void forward_impl_training(const Variables &inputs,
                                     const Variables &outputs);
  virtual void forward_impl_inference(const Variables &inputs,
                                      const Variables &outputs);
  virtual void backward_impl(const Variables &inputs, const Variables &outputs,
                             const vector<bool> &propagate_down,
                             const vector<bool> &accum);
};
}
#endif
