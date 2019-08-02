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

#include <nbla/array.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>
#include <nbla/cuda/cudnn/function/rnn.hpp>
#include <nbla/variable.hpp>

#include <array>
#include <random>

namespace nbla {

template <typename ARRAY>
typename ARRAY::value_type array_product(const ARRAY &arr) {
  typename ARRAY::value_type p = 1;
  for (auto a : arr) {
    p *= a;
  }
  return p;
}

template <typename T>
__global__ void kernel_forward_copy_weights(size_t size, const T *weight,
                                            T *param, int j_stride,
                                            int input_dim) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    int stride;
    stride = (i / input_dim) * j_stride;
    param[i] = weight[i + stride];
  }
}

template <typename T>
__global__ void kernel_forward_copy_bias(size_t size, const T *bias, T *param) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { param[i] = bias[i]; }
}

template <typename T>
__global__ void kernel_accumulate_x_and_h(size_t size, const T *d_ptr, T *d) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { d[i] += d_ptr[i]; }
}

template <typename T>
__global__ void kernel_backward_copy_weights(size_t size, T *g_weight,
                                             T *g_param, int j_stride,
                                             int input_dim, bool accum) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    int stride;
    stride = (i / input_dim) * j_stride;
    if (accum) {
      g_weight[i + stride] += g_param[i];
    } else {
      g_weight[i + stride] = g_param[i];
    }
  }
}

template <typename T>
__global__ void kernel_backward_copy_bias(size_t size, T *g_bias, T *g_param,
                                          bool accum) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    if (accum) {
      g_bias[i] += g_param[i];
    } else {
      g_bias[i] = g_param[i];
    }
  }
}

template <typename T>
void RNNCudaCudnn<T>::copy_weight_bias_to_params(Tcu *params, const Tcu *w_init,
                                                 const Tcu *weight,
                                                 const Tcu *bias,
                                                 bool weight_exists,
                                                 bool bias_exists) {
  for (int64_t layer_id = 0; layer_id < this->num_layers_ * num_directions_;
       layer_id++) {
    for (int64_t lin_layer_id = 0; lin_layer_id < num_lin_layers_;
         lin_layer_id++) {
      int param_index = layer_id * num_lin_layers_ + lin_layer_id;
      int inweight_offset = 0;
      if (layer_id / num_directions_ == 0) {
        if (lin_layer_id == 0) {
          inweight_offset =
              layer_id * (input_dim_ + hidden_size_) * hidden_size_;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_forward_copy_weights<Tcu>),
              weight_offsets_[param_index].second, w_init + inweight_offset,
              params + weight_offsets_[param_index].first / sizeof(T),
              hidden_size_, input_dim_);
        } else {
          inweight_offset =
              layer_id * (input_dim_ + hidden_size_) * hidden_size_ +
              input_dim_;
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_forward_copy_weights<Tcu>),
              weight_offsets_[param_index].second, w_init + inweight_offset,
              params + weight_offsets_[param_index].first / sizeof(T),
              input_dim_, hidden_size_)
        }
      } else {
        if (lin_layer_id == 0) {
          inweight_offset = (layer_id - num_directions_) *
                            (num_directions_ * hidden_size_ + hidden_size_) *
                            hidden_size_;
          if (this->num_layers_ > 1 && weight_exists) {
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
                (kernel_forward_copy_weights<Tcu>),
                weight_offsets_[param_index].second, weight + inweight_offset,
                params + weight_offsets_[param_index].first / sizeof(T),
                hidden_size_, num_directions_ * hidden_size_);
          }
        } else {
          inweight_offset =
              (layer_id - num_directions_) *
                  (num_directions_ * hidden_size_ + hidden_size_) *
                  hidden_size_ +
              num_directions_ * hidden_size_;
          if (this->num_layers_ > 1 && weight_exists) {
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
                (kernel_forward_copy_weights<Tcu>),
                weight_offsets_[param_index].second, weight + inweight_offset,
                params + weight_offsets_[param_index].first / sizeof(T),
                num_directions_ * hidden_size_, hidden_size_);
          }
        }
      }
      if (bias && lin_layer_id == 0) { // copy only when lin_layer_id = 0
        if (bias_exists) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_forward_copy_bias<Tcu>),
              bias_offsets_[param_index].second,
              bias + (layer_id * hidden_size_),
              params + bias_offsets_[param_index].first / sizeof(T));
        }
      }
    }
  }
}

template <typename T>
void RNNCudaCudnn<T>::copy_params_to_gradients(
    Tcu *params, Tcu *w_init, Tcu *weight, Tcu *bias, bool w_init_accum,
    bool w_accum, bool b_accum, bool w_init_propagate, bool w_propagate,
    bool b_propagate) {
  for (int64_t layer_id = 0; layer_id < this->num_layers_ * num_directions_;
       layer_id++) {
    for (int64_t lin_layer_id = 0; lin_layer_id < num_lin_layers_;
         lin_layer_id++) {
      int param_index = layer_id * num_lin_layers_ + lin_layer_id;
      int inweight_offset = 0;
      if (layer_id / num_directions_ == 0) {
        if (lin_layer_id == 0) {
          inweight_offset =
              layer_id * (input_dim_ + hidden_size_) * hidden_size_; //= 0
          if (w_init_propagate) {
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
                (kernel_backward_copy_weights<Tcu>),
                weight_offsets_[param_index].second, w_init + inweight_offset,
                params + weight_offsets_[param_index].first / sizeof(T),
                hidden_size_, input_dim_, w_init_accum);
          }

        } else {
          inweight_offset =
              layer_id * (input_dim_ + hidden_size_) * hidden_size_ +
              input_dim_;

          if (w_init_propagate) {
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
                (kernel_backward_copy_weights<Tcu>),
                weight_offsets_[param_index].second, w_init + inweight_offset,
                params + weight_offsets_[param_index].first / sizeof(T),
                input_dim_, hidden_size_, w_init_accum);
          }
        }
      } else {
        if (lin_layer_id == 0) {
          inweight_offset = (layer_id - num_directions_) *
                            (num_directions_ * hidden_size_ + hidden_size_) *
                            hidden_size_;

          if (w_propagate) {
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
                (kernel_backward_copy_weights<Tcu>),
                weight_offsets_[param_index].second, weight + inweight_offset,
                params + weight_offsets_[param_index].first / sizeof(T),
                hidden_size_, num_directions_ * hidden_size_, w_accum);
          }
        } else {
          inweight_offset =
              (layer_id - num_directions_) *
                  (num_directions_ * hidden_size_ + hidden_size_) *
                  hidden_size_ +
              num_directions_ * hidden_size_;

          if (w_propagate) {
            NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
                (kernel_backward_copy_weights<Tcu>),
                weight_offsets_[param_index].second, weight + inweight_offset,
                params + weight_offsets_[param_index].first / sizeof(T),
                num_directions_ * hidden_size_, hidden_size_, w_accum);
          }
        }
      }
      if (bias && lin_layer_id == 0) { // copy only when lin_layer_id = 0
        if (b_propagate) {
          NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(
              (kernel_backward_copy_bias<Tcu>),
              bias_offsets_[param_index].second,
              bias + (layer_id * hidden_size_),
              params + bias_offsets_[param_index].first / sizeof(T), b_accum);
        }
      }
    }
  }
}

template <typename T>
void RNNCudaCudnn<T>::setup_impl(const Variables &inputs,
                                 const Variables &outputs) {

  // Create x descriptors and y descriptors by resizing
  // Set descriptors

  cuda_set_device(this->device_);
  auto cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(this->device_);

  Shape_t inshape = inputs[0]->shape();
  Shape_t hshape = inputs[1]->shape();
  Shape_t outshape = outputs[0]->shape();

  // Check input dimensions
  NBLA_CHECK(inputs[0]->ndim() == 3, error_code::value,
             "Input x must be a 3 dimensional array with a shape of (steps, "
             "batch_size, input_size).");

  // Get input dimensions
  cudnnDataType_t dt = cudnn_data_type<T>::type();
  seq_len_ = inshape[0];
  int batch_size = inshape[1];
  input_dim_ = inshape[2];
  // Assuming this function takes h as (numLayer, numD, B, M)
  hidden_size_ = hshape[3];
  inputMode = CUDNN_LINEAR_INPUT;
  num_directions_ = this->bidirectional_ ? 2 : 1;
  direction = this->bidirectional_ ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL;

  // Check h shape
  const char *error_msg_h = "Input h must be a 4 dimensional array with a "
                            "shape of (num_layers, num_directions, batch_size, "
                            "hidden_size).";
  NBLA_CHECK(inputs[1]->ndim() == 4, error_code::value, error_msg_h);
  NBLA_CHECK(hshape[0] == this->num_layers_, error_code::value, error_msg_h);
  NBLA_CHECK(hshape[1] == num_directions_, error_code::value, error_msg_h);
  NBLA_CHECK(hshape[2] == batch_size, error_code::value, error_msg_h);

  // Check weight shape at 0th layer
  Shape_t w0_shape = inputs[2]->shape();
  const char *error_msg_w0 = "Input w0 must be a 3 dimensional array with a "
                             "shape of (num_directions, hidden_size, "
                             "input_size + hidden_size).";
  NBLA_CHECK(inputs[2]->ndim() == 3, error_code::value, error_msg_w0);
  NBLA_CHECK(w0_shape[0] == num_directions_, error_code::value, error_msg_w0);
  NBLA_CHECK(w0_shape[1] == hidden_size_, error_code::value, error_msg_w0);
  NBLA_CHECK(w0_shape[2] == hidden_size_ + input_dim_, error_code::value,
             error_msg_w0);

  if (this->nonlinearity_ == "tanh") {
    RNNMode = CUDNN_RNN_TANH;
    num_lin_layers_ = 2;
  } else if (this->nonlinearity_ == "relu") {
    RNNMode = CUDNN_RNN_RELU;
    num_lin_layers_ = 2;
  } else {
    NBLA_ERROR(error_code::not_implemented,
               "Non-linearity must be tanh or relu");
  }

  weight_exists_ = true;
  bias_exists_ = true;
  if (inputs.size() == 3) {
    weight_exists_ = false;
    bias_exists_ = false;
  } else if (inputs.size() == 4) {
    Shape_t opt_shape = inputs[3]->shape();
    if (this->num_layers_ > 1 && opt_shape.size() == 4) {
      bias_exists_ = false;
    } else if (this->num_layers_ > 1 && opt_shape.size() != 4) {
      NBLA_ERROR(error_code::value,
                 "Weight argument must be passed when num_layers > 1");
    } else if (this->num_layers_ == 1 && opt_shape.size() != 3) {
      NBLA_ERROR(error_code::value,
                 "Weight argument cannot be passed when num_layers == 1");
    } else if (this->num_layers_ == 1 && opt_shape.size() == 3) {
      weight_exists_ = false;
    }
  } else if ((inputs.size() > 4) && (this->num_layers_ == 1)) {
    NBLA_ERROR(error_code::value,
               "Weight argument cannot be passed when num_layers == 1");
  }

  // Check weight shape
  if (weight_exists_) {
    Shape_t w_shape = inputs[3]->shape();
    const char *error_msg_w = "Input w must be a 4 dimensional array with a "
                              "shape of (num_layers - 1, num_directions, "
                              "hidden_size, num_directions * hidden_size + "
                              "hidden_size).";
    NBLA_CHECK(inputs[3]->ndim() == 4, error_code::value, error_msg_w);
    NBLA_CHECK(w_shape[0] == this->num_layers_ - 1, error_code::value,
               error_msg_w);
    NBLA_CHECK(w_shape[1] == num_directions_, error_code::value, error_msg_w);
    NBLA_CHECK(w_shape[2] == hidden_size_, error_code::value, error_msg_w);
    NBLA_CHECK(w_shape[3] == num_directions_ * hidden_size_ + hidden_size_,
               error_code::value, error_msg_w);
  }

  // Check bias shape
  if (bias_exists_) {
    const int b_index = weight_exists_ ? 4 : 3;
    Shape_t b_shape = inputs[b_index]->shape();
    const char *error_msg_b = "Input b must be a 3 dimensional array with a "
                              "shape of (num_layers, num_directions, "
                              "hidden_size).";
    NBLA_CHECK(inputs[b_index]->ndim() == 3, error_code::value, error_msg_b);
    NBLA_CHECK(b_shape[0] == this->num_layers_, error_code::value, error_msg_b);
    NBLA_CHECK(b_shape[1] == num_directions_, error_code::value, error_msg_b);
    NBLA_CHECK(b_shape[2] == hidden_size_, error_code::value, error_msg_b);
  }

  // Set X desc
  // xdesc : T * (B, N, 1)
  // x : (T, B, N) row-major
  x_desc_.reset(new WCudnnTensorDescArray(seq_len_));
  for (auto &x : x_desc_->desc_array()) {
    std::array<int, 3> dimA{batch_size, input_dim_, 1};
    std::array<int, 3> strideA{input_dim_, 1, 1};
    NBLA_CUDNN_CHECK(cudnnSetTensorNdDescriptor(x, cudnn_data_type<T>::type(),
                                                dimA.size(), dimA.data(),
                                                strideA.data()));
  }

  // Set hx and hy desc
  // hxDesc : (numLayer * numD, B, M)
  // hx : (numLayer, numD, B, M) row-major >>> or (numD, numLayer, B, M)
  // row-major
  // hyDesc : (numLayer * numD, B, M)
  // hy : (numLayer, numD, B, M) row-major >>> or (numD, numLayer, B, M)
  // row-major
  {
    std::array<int, 3> dimA{this->num_layers_ * num_directions_, batch_size,
                            hidden_size_};
    std::array<int, 3> strideA{batch_size * hidden_size_, hidden_size_, 1};
    NBLA_CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(h_desc_.desc, cudnn_data_type<T>::type(),
                                   dimA.size(), dimA.data(), strideA.data()));
    NBLA_CUDNN_CHECK(
        cudnnSetTensorNdDescriptor(h_n_desc_.desc, cudnn_data_type<T>::type(),
                                   dimA.size(), dimA.data(), strideA.data()));
  }

  // Set Y desc
  // yDesc : T * (B, M * numD, 1)
  // y : (T, B, M, numD) row-major, >>> or (T, B, numD, M)
  y_desc_.reset(new WCudnnTensorDescArray(seq_len_));
  for (auto &y : y_desc_->desc_array()) {
    std::array<int, 3> dimA{batch_size, hidden_size_ * num_directions_, 1};
    std::array<int, 3> strideA{hidden_size_ * num_directions_, 1, 1};
    NBLA_CUDNN_CHECK(cudnnSetTensorNdDescriptor(y, cudnn_data_type<T>::type(),
                                                dimA.size(), dimA.data(),
                                                strideA.data()));
  }

  // Get an RNN algorithm using cudnnGetRNNAlgorithm or cudnnFindRNNAlgorithm.
  // NOTE: find algorithm executes many algorithms exhaustively, and find a best
  // one.

  // Set dropout descriptor
  size_t dropout_stateSize;
  NBLA_CUDNN_CHECK(cudnnDropoutGetStatesSize(cudnn_handle, &dropout_stateSize));
  state_array_ =
      make_shared<CudaCachedArray>(dropout_stateSize, dtypes::BYTE, this->ctx_);
  void *state_ptr = state_array_->pointer<void>();
  std::random_device seed_gen;
  std::default_random_engine engine(seed_gen());
  std::uniform_int_distribution<> dist(0, 999);
  NBLA_CUDNN_CHECK(cudnnSetDropoutDescriptor(dropout_desc_.desc, cudnn_handle,
                                             this->dropout_, state_ptr,
                                             dropout_stateSize, dist(engine)));

// Set RNN descriptor.
#if CUDNN_VERSION >= 7000
  NBLA_CUDNN_CHECK(cudnnSetRNNDescriptor_v5(
      rnn_desc_.desc, hidden_size_, this->num_layers_, dropout_desc_.desc,
      inputMode, direction, RNNMode, dt));
#else
  NBLA_CUDNN_CHECK(cudnnSetRNNDescriptor(rnn_desc_.desc, hidden_size_,
                                         this->num_layers_, dropout_desc_.desc,
                                         inputMode, direction, RNNMode, dt));
#endif

  // Get workspace size and reserve size
  NBLA_CUDNN_CHECK(cudnnGetRNNWorkspaceSize(cudnn_handle, rnn_desc_.desc,
                                            seq_len_, x_desc_->data(),
                                            &workspace_size_));
  if (this->training_) {
    NBLA_CUDNN_CHECK(
        cudnnGetRNNTrainingReserveSize(cudnn_handle, rnn_desc_.desc, seq_len_,
                                       x_desc_->data(), &reserve_size_));
  }

  // Get number of pararameters both in bytes and in elements.
  NBLA_CUDNN_CHECK(cudnnGetRNNParamsSize(cudnn_handle, rnn_desc_.desc,
                                         x_desc_->data()[0],
                                         &params_size_in_bytes_, dt));
  total_params_ = params_size_in_bytes_ / sizeof(T);
  // Set params descriptor
  {
    std::array<int, 3> filter_dims{(int)total_params_, 1, 1};
    NBLA_CUDNN_CHECK(cudnnSetFilterNdDescriptor(
        params_desc_.desc, cudnn_data_type<T>::type(), CUDNN_TENSOR_NCHW, 3,
        filter_dims.data()));
  }

  // Calculate address corerspondences between input parameters (weights and
  // biases) and flattened parameters buffer.
  // weight : [H, I+H]
  // bias : [H]

  // Temporary buffer. This is used only for getting address offsets of matrix
  // and biases from the head of the params pointer.
  CudaCachedArray params_array(params_size_in_bytes_, dtypes::BYTE, this->ctx_);
  Tcu *params = params_array.pointer<Tcu>();

  weight_offsets_.clear();
  bias_offsets_.clear();
  WCudnnFilterDesc lin_layer_mat_desc;
  for (int64_t layer_id = 0; layer_id < this->num_layers_ * num_directions_;
       layer_id++) {
    for (int64_t lin_layer_id = 0; lin_layer_id < num_lin_layers_;
         lin_layer_id++) {
      void *matrix_pointer;
      int nb_dims;
      cudnnDataType_t data_type;
      cudnnTensorFormat_t format;
      std::array<int, 3> dim;
      NBLA_CUDNN_CHECK(cudnnGetRNNLinLayerMatrixParams(
          cudnn_handle, rnn_desc_.desc, layer_id, x_desc_->data()[0],
          params_desc_.desc, params, lin_layer_id, lin_layer_mat_desc.desc,
          &matrix_pointer));
      NBLA_CUDNN_CHECK(cudnnGetFilterNdDescriptor(lin_layer_mat_desc.desc, 3,
                                                  &data_type, &format, &nb_dims,
                                                  dim.data()));

      // Size of the weight matrix can be obtained by a product of dim elements.
      int weight_size = array_product(dim);
      weight_offsets_.push_back(
          {intptr_t(matrix_pointer) - intptr_t(params), weight_size});
      // Get an address pointer of a bias vector corresponding to layer_id and
      // linear_id, and get its size.
      NBLA_CUDNN_CHECK(cudnnGetRNNLinLayerBiasParams(
          cudnn_handle, rnn_desc_.desc, layer_id, x_desc_->data()[0],
          params_desc_.desc, params, lin_layer_id, lin_layer_mat_desc.desc,
          &matrix_pointer));
      NBLA_CUDNN_CHECK(cudnnGetFilterNdDescriptor(lin_layer_mat_desc.desc, 3,
                                                  &data_type, &format, &nb_dims,
                                                  dim.data()));
      // Size of the bias vector can be obtained by a product of dim elements.

      int bias_size = array_product(dim);
      bias_offsets_.push_back(
          {intptr_t(matrix_pointer) - intptr_t(params), bias_size});
    }
  }

  // Set output shapes
  outputs[0]->reshape({seq_len_, batch_size, num_directions_ * hidden_size_},
                      true);
  outputs[1]->reshape(inputs[1]->shape(), true);
}

template <typename T>
void RNNCudaCudnn<T>::forward_impl(const Variables &inputs,
                                   const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  if (this->training_) { // Training mode.
    forward_impl_training(inputs, outputs);
  } else { // Testing mode.
    forward_impl_inference(inputs, outputs);
  }
}

template <typename T>
void RNNCudaCudnn<T>::forward_impl_training(const Variables &inputs,
                                            const Variables &outputs) {

  cuda_set_device(this->device_);
  auto cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(this->device_);

  // Inputs and outputs
  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *h = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *w_init = inputs[2]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *weight{nullptr};
  const Tcu *bias{nullptr};
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_);
  Tcu *h_n = outputs[1]->cast_data_and_get_pointer<Tcu>(this->ctx_);
  if (inputs.size() == 4) {
    if (weight_exists_) {
      weight = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    } else if (bias_exists_) {
      bias = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    }
  }
  if (inputs.size() > 4) {
    weight = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    bias = inputs[4]->get_data_pointer<Tcu>(this->ctx_);
  }

  // Create flattened weight buffer.
  CudaCachedArray params_array(params_size_in_bytes_, dtypes::BYTE, this->ctx_);
  params_array.zero(); // Initialize params with 0
  Tcu *params = params_array.pointer<Tcu>();

  this->copy_weight_bias_to_params(params, w_init, weight, bias, weight_exists_,
                                   bias_exists_);

  shared_ptr<CudaCachedArray> mem_workspace{nullptr};
  if (workspace_size_) {
    mem_workspace.reset(
        new CudaCachedArray(workspace_size_, dtypes::BYTE, this->ctx_));
  }
  if (mem_reservespace_) {
    NBLA_CHECK(mem_reservespace_->size() == reserve_size_, error_code::value,
               "reserve_size_ is inconsistent with the previously set "
               "reservespace size.");
  }
  mem_reservespace_.reset(
      new CudaCachedArray(reserve_size_, dtypes::BYTE, this->ctx_));

  auto alpha = get_cudnn_scalar_arg<T>(1);
  auto beta = get_cudnn_scalar_arg<T>(0);
  NBLA_CUDNN_CHECK(cudnnRNNForwardTraining(
      cudnn_handle, rnn_desc_.desc, seq_len_, x_desc_->data(), x, h_desc_.desc,
      h, c_x_desc_.desc, NULL, params_desc_.desc, params, y_desc_->data(), y,
      h_n_desc_.desc, h_n, c_y_desc_.desc, NULL, mem_workspace->pointer(),
      workspace_size_, mem_reservespace_->pointer(), reserve_size_));
}

template <typename T>
void RNNCudaCudnn<T>::forward_impl_inference(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(this->device_);
  auto cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(this->device_);

  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *h = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *w_init = inputs[2]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *weight{nullptr};
  const Tcu *bias{nullptr};
  Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_);
  Tcu *h_n = outputs[1]->cast_data_and_get_pointer<Tcu>(this->ctx_);
  if (inputs.size() == 4) {
    if (weight_exists_) {
      weight = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    } else if (bias_exists_) {
      bias = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    }
  }
  if (inputs.size() > 4) {
    weight = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    bias = inputs[4]->get_data_pointer<Tcu>(this->ctx_);
  }

  // Create flattened weight buffer.
  CudaCachedArray params_array(params_size_in_bytes_, dtypes::BYTE, this->ctx_);
  params_array.zero(); // Initialize params with 0
  Tcu *params = params_array.pointer<Tcu>();

  this->copy_weight_bias_to_params(params, w_init, weight, bias, weight_exists_,
                                   bias_exists_);

  shared_ptr<CudaCachedArray> mem_workspace{nullptr};
  if (workspace_size_) {
    mem_workspace.reset(
        new CudaCachedArray(workspace_size_, dtypes::BYTE, this->ctx_));
  }

  NBLA_CUDNN_CHECK(cudnnRNNForwardInference(
      cudnn_handle, rnn_desc_.desc, seq_len_, x_desc_->data(), x, h_desc_.desc,
      h, c_x_desc_.desc, NULL, params_desc_.desc, params, y_desc_->data(), y,
      h_n_desc_.desc, h_n, c_y_desc_.desc, NULL,
      mem_workspace ? mem_workspace->pointer() : nullptr, workspace_size_));
}

template <typename T>
void RNNCudaCudnn<T>::backward_impl(const Variables &inputs,
                                    const Variables &outputs,
                                    const vector<bool> &propagate_down,
                                    const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] || propagate_down[2] ||
        (inputs.size() > 3 && propagate_down[3]) ||
        (inputs.size() > 4 && propagate_down[4]))) {
    return;
  }

  NBLA_CHECK(this->training_, error_code::value,
             "Backward is called for training only.");
  NBLA_CHECK(mem_reservespace_, error_code::value,
             "Reserve space should be allocated memory space.");
  NBLA_CHECK(mem_reservespace_->size() == reserve_size_, error_code::value,
             "reserve_size_ is inconsistent with the previously set "
             "reservespace size.");

  if (inputs.size() > 4 && propagate_down[4]) {
    NBLA_CHECK(propagate_down[2] == propagate_down[3], error_code::value,
               "If bias is backpropagated, so should weights.");
  }

  cuda_set_device(this->device_);
  auto cudnn_handle =
      SingletonManager::get<CudnnHandleManager>()->handle(this->device_);

  const Tcu *x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *h = inputs[1]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *w_init = inputs[2]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *weight{nullptr};
  const Tcu *bias{nullptr};
  const Tcu *g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);
  const Tcu *g_h_n = outputs[1]->get_grad_pointer<Tcu>(this->ctx_);
  if (inputs.size() == 4) {
    if (weight_exists_) {
      weight = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    } else if (bias_exists_) {
      bias = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    } else {
      NBLA_CHECK(this->training_, error_code::value,
                 "4th input is neither weight nor bias.");
    }
  }
  if (inputs.size() > 4) {
    weight = inputs[3]->get_data_pointer<Tcu>(this->ctx_);
    bias = inputs[4]->get_data_pointer<Tcu>(this->ctx_);
  }

  const Tcu *y = outputs[0]->get_data_pointer<Tcu>(this->ctx_);
  const Tcu *h_n = outputs[1]->get_data_pointer<Tcu>(this->ctx_);
  Tcu *g_x{nullptr};
  Tcu *g_h{nullptr};
  Tcu *g_w_init{nullptr};
  Tcu *g_weight{nullptr};
  Tcu *g_bias{nullptr};

  CudaCachedArray params_array(params_size_in_bytes_, dtypes::BYTE, this->ctx_);
  CudaCachedArray g_params_array(params_size_in_bytes_, dtypes::BYTE,
                                 this->ctx_);
  params_array.zero(); // Initialize params with 0
  g_params_array.zero();
  Tcu *params = params_array.pointer<Tcu>();
  Tcu *g_params = g_params_array.pointer<Tcu>();

  this->copy_weight_bias_to_params(params, w_init, weight, bias, weight_exists_,
                                   bias_exists_);

  if (propagate_down[0]) {
    g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  }
  if (propagate_down[1]) {
    g_h = inputs[1]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[1]);
  }
  if (propagate_down[2]) {
    g_w_init = inputs[2]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[2]);
  }
  if (inputs.size() == 4 && propagate_down[3]) {
    if (weight_exists_) {
      g_weight =
          inputs[3]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[3]);
    } else if (bias_exists_) {
      g_bias = inputs[3]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[3]);
    }
  }
  if (inputs.size() == 5 && propagate_down[3]) {
    g_weight = inputs[3]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[3]);
  }
  if (inputs.size() == 5 && propagate_down[4]) {
    g_bias = inputs[4]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[4]);
  }

  shared_ptr<CudaCachedArray> mem_workspace{nullptr};
  if (workspace_size_) {
    mem_workspace.reset(
        new CudaCachedArray(workspace_size_, dtypes::BYTE, this->ctx_));
  }

  shared_ptr<CudaCachedArray> mem_x_accum{nullptr};
  shared_ptr<CudaCachedArray> mem_h_accum{nullptr};
  Tcu *dx_tmp = g_x;
  Tcu *dh_tmp = g_h;

  if (!propagate_down[0] || accum[0]) {
    mem_x_accum.reset(new CudaCachedArray(inputs[0]->size() * sizeof(Tcu),
                                          dtypes::BYTE, this->ctx_));
    dx_tmp = mem_x_accum->pointer<Tcu>();
  }
  if (!propagate_down[1] || accum[1]) {
    mem_h_accum.reset(new CudaCachedArray(inputs[1]->size() * sizeof(Tcu),
                                          dtypes::BYTE, this->ctx_));
    dh_tmp = mem_h_accum->pointer<Tcu>();
  }

  NBLA_CUDNN_CHECK(cudnnRNNBackwardData(
      cudnn_handle, rnn_desc_.desc, seq_len_, y_desc_->data(), y,
      y_desc_->data(), g_y, h_n_desc_.desc, g_h_n, c_y_desc_.desc, NULL,
      params_desc_.desc, params, h_desc_.desc, h, c_x_desc_.desc, NULL,
      x_desc_->data(), dx_tmp, h_desc_.desc, dh_tmp, c_x_desc_.desc, NULL,
      mem_workspace->pointer(), workspace_size_, mem_reservespace_->pointer(),
      reserve_size_));

  if (propagate_down[0] && accum[0]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_accumulate_x_and_h<Tcu>),
                                   inputs[0]->size(), dx_tmp, g_x);
  }
  if (propagate_down[1] && accum[1]) {
    NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_accumulate_x_and_h<Tcu>),
                                   inputs[1]->size(), dh_tmp, g_h);
  }

  if (propagate_down[2] || (inputs.size() > 3 && propagate_down[3]) ||
      (inputs.size() == 5 && propagate_down[4])) {
    NBLA_CUDNN_CHECK(cudnnRNNBackwardWeights(
        cudnn_handle, rnn_desc_.desc, seq_len_, x_desc_->data(), x,
        h_desc_.desc, h, y_desc_->data(), y, mem_workspace->pointer(),
        workspace_size_, params_desc_.desc, g_params,
        mem_reservespace_->pointer(), reserve_size_));
  }

  bool w_init_accum = false;
  bool w_accum = false;
  bool b_accum = false;
  bool w_prop = false;
  bool b_prop = false;
  if (propagate_down[2] && accum[2]) {
    w_init_accum = true;
  }
  if (inputs.size() > 3 && propagate_down[3]) {
    if (inputs.size() == 4 && weight_exists_) {
      w_prop = true;
      if (accum[3]) {
        w_accum = true;
      }
    } else if (inputs.size() == 4 && bias_exists_) {
      b_prop = true;
      if (accum[3]) {
        b_accum = true;
      }
    } else {
      w_prop = true;
      if (accum[3]) {
        w_accum = true;
      }
    }
  }
  if (inputs.size() == 5 && propagate_down[4]) {
    b_prop = true;
    if (accum[4]) {
      b_accum = true;
    }
  }

  this->copy_params_to_gradients(g_params, g_w_init, g_weight, g_bias,
                                 w_init_accum, w_accum, b_accum,
                                 propagate_down[2], w_prop, b_prop);
}
}
