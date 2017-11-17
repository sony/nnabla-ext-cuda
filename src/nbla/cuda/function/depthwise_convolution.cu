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

#include <nbla/cuda/function/depthwise_convolution.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/cuda/utils/block_reduce.cuh>
#include <nbla/singleton_manager.hpp>

namespace nbla {

namespace depthwise_convolution_cuda {

template <typename T, int K>
__global__ void
forward_kernel(const T *input_data, T *output_data, const T *weight_data,
               const T *bias_data, const int output_data_size,
               const int2 sample, const int2 outmap, const int kernel,
               const int stride, const int padding, const int dilation) {
  // sample/outmap (x, y) == (length, channels)
  // outmap.y == sample.y * multiplier
  const auto &sample_channels = sample.y;
  const auto &outmap_channels = outmap.y;
  const auto &sample_size = sample.x;
  const auto &outmap_size = outmap.x;

  const int KERNEL = (K != 0) ? K : kernel;

  NBLA_CUDA_KERNEL_LOOP(output_offset, output_data_size) {
    const int samp = output_offset / (outmap_channels * outmap_size);
    const int chan = (output_offset / outmap_size) % outmap_channels;
    const int outmap_x = output_offset % outmap_size;
    const int sample_base = (samp * sample_channels + chan) * sample_size;

    T value = (bias_data != nullptr) ? bias_data[chan] : 0;
    int weight_offset = chan * kernel;

#pragma unroll
    for (int k = 0; k < KERNEL; ++k) {
      const int sample_x = -padding + outmap_x * stride + k * dilation;

      if ((sample_x >= 0) && (sample_x < sample_size)) {
        const int input_offset = sample_base + sample_x;
        value += weight_data[weight_offset] * input_data[input_offset];
      }
      ++weight_offset;
    }
    output_data[output_offset] = value;
  }
}

template <typename T, int K>
__global__ void
forward_kernel(const T *input_data, T *output_data, const T *weight_data,
               const T *bias_data, const int output_data_size,
               const int3 sample, const int3 outmap, const int2 kernel,
               const int2 stride, const int2 padding, const int2 dilation) {
  // sample/outmap (x, y, z) == (width, height, channels)
  // outmap.z == sample.z * multiplier
  // sample/outmap/kernel size == width * height
  const auto &sample_channels = sample.z;
  const auto &outmap_channels = outmap.z;

  const int sample_size = sample.x * sample.y;
  const int outmap_size = outmap.x * outmap.y;

  const int KERNEL_X = (K != 0) ? K : kernel.x;
  const int KERNEL_Y = (K != 0) ? K : kernel.y;
  const int kernel_size = KERNEL_X * KERNEL_Y;

  NBLA_CUDA_KERNEL_LOOP(output_offset, output_data_size) {
    const int samp = output_offset / (outmap_channels * outmap_size);
    const int chan = (output_offset / outmap_size) % outmap_channels;
    const int outmap_y = (output_offset / outmap.x) % outmap.y;
    const int outmap_x = output_offset % outmap.x;
    const int sample_base = (samp * sample_channels + chan) * sample_size;

    T value = (bias_data != nullptr) ? bias_data[chan] : 0;
    int weight_offset = chan * kernel_size;

#pragma unroll
    for (int k_y = 0; k_y < KERNEL_Y; ++k_y) {
#pragma unroll
      for (int k_x = 0; k_x < KERNEL_X; ++k_x) {
        const int sample_y =
            -padding.y + outmap_y * stride.y + k_y * dilation.y;
        const int sample_x =
            -padding.x + outmap_x * stride.x + k_x * dilation.x;

        if ((sample_y >= 0) && (sample_y < sample.y) && (sample_x >= 0) &&
            (sample_x < sample.x)) {
          const int input_offset = sample_base + sample_y * sample.x + sample_x;
          value += weight_data[weight_offset] * input_data[input_offset];
        }
        ++weight_offset;
      }
    }
    output_data[output_offset] = value;
  }
}

template <typename T, int K>
__global__ void backprop_input(T *input_grad, const T *output_grad,
                               const T *weight_data, const int input_data_size,
                               const int2 sample, const int2 outmap,
                               const int kernel, const int stride,
                               const int padding, const int dilation) {
  // sample/outmap (x, y) == (length, channels)
  // outmap.y == sample.y * multiplier
  const auto &sample_channels = sample.y;
  const auto &outmap_channels = outmap.y;
  const auto &sample_size = sample.x;
  const auto &outmap_size = outmap.x;

  const int KERNEL = (K != 0) ? K : kernel;

  NBLA_CUDA_KERNEL_LOOP(input_offset, input_data_size) {
    const int samp = input_offset / (sample_channels * sample_size);
    const int chan = (input_offset / sample_size) % sample_channels;
    const int sample_x = input_offset % sample_size;
    const int outmap_base = (samp * outmap_channels + chan) * outmap_size;

    T value = 0;
    int weight_offset = chan * kernel;

#pragma unroll
    for (int k = 0; k < KERNEL; ++k) {
      int outmap_x = sample_x + padding - k * dilation;

      const int outmap_x_div_stride = outmap_x / stride;

      if (outmap_x - outmap_x_div_stride * stride == 0) {
        outmap_x = outmap_x_div_stride;

        if ((outmap_x >= 0) && (outmap_x < outmap.x)) {
          const int output_offset = outmap_base + outmap_x;
          value += weight_data[weight_offset] * output_grad[output_offset];
        }
      }
      ++weight_offset;
    }
    input_grad[input_offset] += value;
  }
}

template <typename T, int K>
__global__ void backprop_input(T *input_grad, const T *output_grad,
                               const T *weight_data, const int input_data_size,
                               const int3 sample, const int3 outmap,
                               const int2 kernel, const int2 stride,
                               const int2 padding, const int2 dilation) {
  // sample/outmap (x, y, z) == (width, height, channels)
  // outmap.z == sample.z * multiplier
  // sample/outmap/kernel size == width * height
  const auto &sample_channels = sample.z;
  const auto &outmap_channels = outmap.z;
  const auto sample_size = sample.x * sample.y;
  const auto outmap_size = outmap.x * outmap.y;

  const int KERNEL_X = (K != 0) ? K : kernel.x;
  const int KERNEL_Y = (K != 0) ? K : kernel.y;
  const auto kernel_size = kernel.x * kernel.y;

  NBLA_CUDA_KERNEL_LOOP(input_offset, input_data_size) {
    const int samp = input_offset / (sample_channels * sample_size);
    const int chan = (input_offset / sample_size) % sample_channels;
    const int sample_y = (input_offset / sample.x) % sample.y;
    const int sample_x = input_offset % sample.x;
    const int outmap_base = (samp * outmap_channels + chan) * outmap_size;

    T value = 0;
    int weight_offset = chan * kernel_size;

#pragma unroll
    for (int k_y = 0; k_y < KERNEL_Y; ++k_y) {
#pragma unroll
      for (int k_x = 0; k_x < KERNEL_X; ++k_x) {
        int outmap_y = sample_y + padding.y - k_y * dilation.y;
        int outmap_x = sample_x + padding.x - k_x * dilation.x;

        const int outmap_y_div_stride = outmap_y / stride.y;
        const int outmap_x_div_stride = outmap_x / stride.x;

        if ((outmap_y - outmap_y_div_stride * stride.y == 0) &&
            (outmap_x - outmap_x_div_stride * stride.x == 0)) {
          outmap_y = outmap_y_div_stride;
          outmap_x = outmap_x_div_stride;

          if ((outmap_y >= 0) && (outmap_y < outmap.y) && (outmap_x >= 0) &&
              (outmap_x < outmap.x)) {
            const int output_offset =
                outmap_base + outmap_y * outmap.x + outmap_x;
            value += weight_data[weight_offset] * output_grad[output_offset];
          }
        }
        ++weight_offset;
      }
    }
    input_grad[input_offset] += value;
  }
}

template <typename T>
__global__ void
backprop_weights(const T *output_grad, const T *input_data, T *weight_grad,
                 T *bias_grad, const int batch_size, const int2 sample,
                 const int2 outmap, const int kernel, const int stride,
                 const int padding, const int dilation) {
  const auto &sample_channels = sample.y;
  const auto &outmap_channels = outmap.y;
  const auto &sample_size = sample.x;
  const auto &outmap_size = outmap.x;

  // gridDim.x == blocks == outmap_channels * kernel_size == total weights
  // blockDim.x == threads == min(batch_size * warpSize, 256)
  // each block computes one weight value

  const int k_x = blockIdx.x % kernel;
  const int chan = blockIdx.x / kernel;
  const int samp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int warps = blockDim.x / warpSize;

  const int sample_stride = sample_channels * sample_size;
  const int outmap_stride = outmap_channels * outmap_size;
  const int sample_offset = chan * sample_size;
  const int outmap_offset = chan * outmap_size;
  const int sample_skip_x = k_x * dilation;

  T weight_grad_value = 0;
  T bias_grad_value = 0;

  // Each thread block (of outmap-channels * kernel-size blocks)
  // computes a weight gradient. It also computes a bias gradient
  // even if that is done kernel-size times more often than
  // needed (but we won't gain with a conditional).
  for (int s = samp; s < batch_size; s += warps) {
    const int sample_base = s * sample_stride + sample_offset;
    const int outmap_base = s * outmap_stride + outmap_offset;

    for (int idx = lane; idx < outmap_size; idx += warpSize) {
      const int outmap_x = idx % outmap.x;
      const int sample_x = (outmap_x * stride) + sample_skip_x - padding;
      const T output_grad_value = output_grad[outmap_base + idx];

      if ((sample_x >= 0) && (sample_x < sample.x)) {
        int input_offset = sample_base + sample_x;
        weight_grad_value += input_data[input_offset] * output_grad_value;
      }
      bias_grad_value += output_grad_value;
    }
  }
  __syncthreads();

  // Accumulate thread local gradients.
  weight_grad_value = blockReduceSum(weight_grad_value);
  bias_grad_value = blockReduceSum(bias_grad_value);

  // The first thread has the accumulated gradients. The bias
  // gradient is needed only once per output channel.
  if (threadIdx.x == 0) {
    const int weight_offset = k_x + kernel * chan;
    weight_grad[weight_offset] += weight_grad_value;
    if (bias_grad && (blockIdx.x % kernel == 0)) {
      bias_grad[chan] += bias_grad_value;
    }
  }
}

template <typename T>
__global__ void
backprop_weights(const T *output_grad, const T *input_data, T *weight_grad,
                 T *bias_grad, const int batch_size, const int3 sample,
                 const int3 outmap, const int2 kernel, const int2 stride,
                 const int2 padding, const int2 dilation) {
  const auto &sample_channels = sample.z;
  const auto &outmap_channels = outmap.z;
  const auto sample_size = sample.x * sample.y;
  const auto outmap_size = outmap.x * outmap.y;
  const auto kernel_size = kernel.x * kernel.y;

  // gridDim.x == blocks == outmap_channels * kernel_size == total weights
  // blockDim.x == threads == min(batch_size * warpSize, 256)
  // each block computes one weight value

  const int k_x = blockIdx.x % kernel.x;
  const int k_y = (blockIdx.x / kernel.x) % kernel.y;
  const int chan = blockIdx.x / kernel_size;
  const int samp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int warps = blockDim.x / warpSize;

  const int sample_stride = sample_channels * sample_size;
  const int outmap_stride = outmap_channels * outmap_size;
  const int sample_offset = chan * sample_size;
  const int outmap_offset = chan * outmap_size;
  const int sample_skip_x = k_x * dilation.x;
  const int sample_skip_y = k_y * dilation.y;

  T weight_grad_value = 0;
  T bias_grad_value = 0;

  // Each thread block (of outmap-channels * kernel-size blocks)
  // computes a weight gradient. It also computes a bias gradient
  // even if that is done kernel-size times more often than
  // needed (but we won't gain with a conditional).
  for (int s = samp; s < batch_size; s += warps) {
    const int sample_base = s * sample_stride + sample_offset;
    const int outmap_base = s * outmap_stride + outmap_offset;

    for (int idx = lane; idx < outmap_size; idx += warpSize) {
      const int outmap_x = idx % outmap.x;
      const int outmap_y = idx / outmap.x;
      const int sample_x = (outmap_x * stride.x) + sample_skip_x - padding.x;
      const int sample_y = (outmap_y * stride.y) + sample_skip_y - padding.y;
      const T output_grad_value = output_grad[outmap_base + idx];

      if ((sample_x >= 0) && (sample_x < sample.x) && (sample_y >= 0) &&
          (sample_y < sample.y)) {
        int input_offset = sample_base + (sample_y * sample.x) + sample_x;
        weight_grad_value += input_data[input_offset] * output_grad_value;
      }
      bias_grad_value += output_grad_value;
    }
  }
  __syncthreads();

  // Accumulate thread local gradients.
  weight_grad_value = blockReduceSum(weight_grad_value);
  bias_grad_value = blockReduceSum(bias_grad_value);

  // The first thread has the accumulated gradients. The bias
  // gradient is needed only once per output channel.
  if (threadIdx.x == 0) {
    const int weight_offset = chan * kernel_size + k_y * kernel.x + k_x;
    weight_grad[weight_offset] += weight_grad_value;
    if (bias_grad && (blockIdx.x % kernel_size == 0)) {
      bias_grad[chan] += bias_grad_value;
    }
  }
}

} // namespace depthwise_convolution_cuda

using namespace depthwise_convolution_cuda;

template <typename T>
void DepthwiseConvolutionCuda<T>::setup_impl(const Variables &inputs,
                                             const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  DepthwiseConvolution<T>::setup_impl(inputs, outputs);

  input_data_size_ = inputs[0]->size();
  output_data_size_ = outputs[0]->size();

  NBLA_CHECK(inputs[1]->size() <= 65536, error_code::unclassified,
             "GPU implementation limit reached: output-channels x filter-size "
             "can not be more than 65536.");  // see implementation note

  if (this->spatial_dims_ == 1) {
    sample_1d_ = make_int2(this->spatial_shape_i_[0], this->channels_i_);
    outmap_1d_ = make_int2(this->spatial_shape_o_[0], this->channels_i_);
    kernel_1d_ = this->kernel_[0];
    stride_1d_ = this->stride_[0];
    padding_1d_ = this->pad_[0];
    dilation_1d_ = this->dilation_[0];
  } else {
    sample_2d_ = make_int3(this->spatial_shape_i_[1], this->spatial_shape_i_[0],
                           this->channels_i_);
    outmap_2d_ = make_int3(this->spatial_shape_o_[1], this->spatial_shape_o_[0],
                           this->channels_i_);
    kernel_2d_ = make_int2(this->kernel_[1], this->kernel_[0]);
    stride_2d_ = make_int2(this->stride_[1], this->stride_[0]);
    padding_2d_ = make_int2(this->pad_[1], this->pad_[0]);
    dilation_2d_ = make_int2(this->dilation_[1], this->dilation_[0]);
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, std::stoi(this->ctx_.device_id));
  max_threads_per_block_ = prop.maxThreadsPerBlock;
  warp_size_ = prop.warpSize;
}

template <typename T>
void DepthwiseConvolutionCuda<T>::forward_impl(const Variables &inputs,
                                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));

  Variable *const input = inputs[0];
  Variable *const weight = inputs[1];
  Variable *const bias = (inputs.size() == 3) ? inputs[2] : nullptr;
  Variable *const output = outputs[0];

  const T *input_data = input->get_data_pointer<T>(this->ctx_);
  const T *weight_data = weight->get_data_pointer<T>(this->ctx_);
  const T *bias_data = (bias) ? bias->get_data_pointer<T>(this->ctx_) : nullptr;
  T *output_data = output->cast_data_and_get_pointer<T>(this->ctx_);

  const int threads = max_threads_per_block_;
  const int blocks = (output_data_size_ + threads - 1) / threads;

  if (this->spatial_dims_ == 1) {
    if (kernel_1d_ == 3) {
      forward_kernel<T, 3><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_1d_, outmap_1d_, kernel_1d_, stride_1d_, padding_1d_,
          dilation_1d_);
    } else if (kernel_1d_ == 5) {
      forward_kernel<T, 5><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_1d_, outmap_1d_, kernel_1d_, stride_1d_, padding_1d_,
          dilation_1d_);
    } else {
      forward_kernel<T, 0><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_1d_, outmap_1d_, kernel_1d_, stride_1d_, padding_1d_,
          dilation_1d_);
    }
  } else {
    if ((kernel_2d_.x == 3) && (kernel_2d_.y == 3)) {
      forward_kernel<T, 3><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_2d_, outmap_2d_, kernel_2d_, stride_2d_, padding_2d_,
          dilation_2d_);
    } else if ((kernel_2d_.x == 5) && (kernel_2d_.y == 5)) {
      forward_kernel<T, 5><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_2d_, outmap_2d_, kernel_2d_, stride_2d_, padding_2d_,
          dilation_2d_);
    } else {
      forward_kernel<T, 0><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_2d_, outmap_2d_, kernel_2d_, stride_2d_, padding_2d_,
          dilation_2d_);
    }
  }
}

template <typename T>
void DepthwiseConvolutionCuda<T>::backward_impl(
    const Variables &inputs, const Variables &outputs,
    const vector<bool> &propagate_down, const vector<bool> &accum) {
  if (!(propagate_down[0] || propagate_down[1] ||
        (inputs.size() == 3 && propagate_down[2]))) {
    return;
  }

  cuda_set_device(std::stoi(this->ctx_.device_id));

  Variable *const input = inputs[0];
  Variable *const weight = inputs[1];
  Variable *const bias = (inputs.size() == 3) ? inputs[2] : nullptr;
  Variable *const output = outputs[0];

  const T *input_data = input->get_data_pointer<T>(this->ctx_);
  const T *weight_data = weight->get_data_pointer<T>(this->ctx_);
  const T *output_grad = output->get_grad_pointer<T>(this->ctx_);

  T *input_grad = nullptr;
  if (propagate_down[0]) {
    if (!accum[0])
      input->grad()->zero(); // before cast!
    input_grad = input->cast_grad_and_get_pointer<T>(this->ctx_);
  }

  T *weight_grad = nullptr;
  if (propagate_down[1]) {
    if (!accum[1])
      weight->grad()->zero(); // before cast!
    weight_grad = weight->cast_grad_and_get_pointer<T>(this->ctx_);
  }

  T *bias_grad = nullptr;
  if (inputs.size() == 3 && propagate_down[2]) {
    if (!accum[2])
      bias->grad()->zero(); // before cast!
    bias_grad = bias->cast_grad_and_get_pointer<T>(this->ctx_);
  }

  if (input_grad) {
    // Compute the input gradient. Each thread of threads times
    // blocks computes one input gradient value.
    const int threads = max_threads_per_block_;
    const int blocks = (input_data_size_ + threads - 1) / threads;
    if (this->spatial_dims_ == 1) {
      if (kernel_1d_ == 3) {
        backprop_input<T, 3><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_1d_,
            outmap_1d_, kernel_1d_, stride_1d_, padding_1d_, dilation_1d_);
      } else if (kernel_1d_ == 5) {
        backprop_input<T, 5><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_1d_,
            outmap_1d_, kernel_1d_, stride_1d_, padding_1d_, dilation_1d_);
      } else {
        backprop_input<T, 0><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_1d_,
            outmap_1d_, kernel_1d_, stride_1d_, padding_1d_, dilation_1d_);
      }
    } else {
      if ((kernel_2d_.x == 3) && (kernel_2d_.y == 3)) {
        backprop_input<T, 3><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_2d_,
            outmap_2d_, kernel_2d_, stride_2d_, padding_2d_, dilation_2d_);
      } else if ((kernel_2d_.x == 5) && (kernel_2d_.y == 5)) {
        backprop_input<T, 5><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_2d_,
            outmap_2d_, kernel_2d_, stride_2d_, padding_2d_, dilation_2d_);
      } else {
        backprop_input<T, 0><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_2d_,
            outmap_2d_, kernel_2d_, stride_2d_, padding_2d_, dilation_2d_);
      }
    }
    NBLA_CUDA_KERNEL_CHECK();
  }

  if (weight_grad) {
    // Compute the weight gradient and optionally the bias gradient
    // (if bias_grad is not a nullptr). The number of thread blocks is
    // the number of filter values multiplied by the number of output
    // channels, thus each block computes one filter value for a
    // specific output channel. For each of batch_size samples we
    // spawn warp_size (usually 32) threads for computation, but no
    // more than max_threads_per_block_ (usually 1024). More samples
    // than (max_threads_per_block_ / warp_size_) will stride loop
    // inside the cuda kernel.
    //
    // TODO: Note that the maximum number of thread blocks for kernel
    // invocation is 2^16 (in 2D case this limit could be reached for
    // example with a 128x128 filter and 1024 output channels). If
    // larger dimensions will ever be required, the blocks could be
    // provided as a dim3 structure using x and y members and the
    // kernel be modified to use blockIdx.x * blockIdx.y for offset
    // calculations.
    const int batch_size = this->outer_size_;
    const int threads = min(batch_size * warp_size_, max_threads_per_block_);
    if (this->spatial_dims_ == 1) {
      const int kernel_size = kernel_1d_;
      const int output_channels = outmap_1d_.y;
      const int blocks = output_channels * kernel_size;
      backprop_weights<T><<<blocks, threads>>>(
          output_grad, input_data, weight_grad, bias_grad, batch_size,
          sample_1d_, outmap_1d_, kernel_1d_, stride_1d_, padding_1d_,
          dilation_1d_);
    } else {
      const int kernel_size = kernel_2d_.x * kernel_2d_.y;
      const int output_channels = outmap_2d_.z;
      const int blocks = output_channels * kernel_size;
      backprop_weights<T><<<blocks, threads>>>(
          output_grad, input_data, weight_grad, bias_grad, batch_size,
          sample_2d_, outmap_2d_, kernel_2d_, stride_2d_, padding_2d_,
          dilation_2d_);
    }
    NBLA_CUDA_KERNEL_CHECK();
  }

  if (bias_grad && !weight_grad) {
    // Normally the bias gradient is computed along with the weight
    // gradient in the same cuda kernel. In the unlikely event that
    // only the bias gradient is requested, we fall back to this
    // slower method.
    if (this->spatial_dims_ == 1) {
      const int outmap_size = outmap_1d_.x;
      const int outmap_channels = outmap_1d_.y;
      const T *ones =
          static_cast<const T *>(SingletonManager::get<NNabla>()->ones(
              outmap_size, get_dtype<T>(), this->ctx_));

      for (int samp = 0; samp < this->outer_size_; ++samp) {
        const int output_offset = samp * outmap_channels * outmap_size;
        cuda_gemv<T>(device_, bias_grad, &output_grad[output_offset],
                     outmap_size, outmap_channels, true, ones, outmap_size,
                     T(1), T(1));
      }
    } else {
      const int outmap_size = outmap_2d_.x * outmap_2d_.y;
      const int outmap_channels = outmap_2d_.z;
      const T *ones =
          static_cast<const T *>(SingletonManager::get<NNabla>()->ones(
              outmap_size, get_dtype<T>(), this->ctx_));

      for (int samp = 0; samp < this->outer_size_; ++samp) {
        const int output_offset = samp * outmap_channels * outmap_size;
        cuda_gemv<T>(device_, bias_grad, &output_grad[output_offset],
                     outmap_size, outmap_channels, true, ones, outmap_size,
                     T(1), T(1));
      }
    }
  }
}

// Template instantiation
template class DepthwiseConvolutionCuda<float>;

} // namespace nbla
