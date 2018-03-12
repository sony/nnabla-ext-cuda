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

#include <nbla/cuda/function/depthwise_deconvolution.hpp>
#include <nbla/cuda/math.hpp>
#include <nbla/cuda/utils/block_reduce.cuh>
#include <nbla/singleton_manager.hpp>

namespace nbla {

namespace depthwise_deconvolution_cuda {

template <typename T, int K>
__global__ void forward_kernel(const T *input_data, T *output_data,
                               const T *weight_data, const T *bias_data,
                               const int output_data_size, const int2 sample,
                               const int2 outmap, const int kernel,
                               const int stride, const int padding,
                               const int dilation, const int divisor) {
  // sample/outmap (x, y) == (length, channels)
  // outmap.y == sample.y / divisor
  const auto &sample_channels = sample.y;
  const auto &outmap_channels = outmap.y;
  const auto &sample_size = sample.x;
  const auto &outmap_size = outmap.x;

  const int kernel_size = (K != 0) ? K : kernel;

  NBLA_CUDA_KERNEL_LOOP(output_offset, output_data_size) {
    const int s = output_offset / (outmap_channels * outmap_size);
    const int outmap_x = output_offset % outmap_size;
    const int outmap_chan = (output_offset / outmap_size) % outmap_channels;
    const int sample_chan = outmap_chan * divisor;
    const int sample_base = (s * sample_channels + sample_chan) * sample_size;
    const int weight_base = sample_chan * kernel;

    T value = bias_data ? bias_data[outmap_chan] : (T)0;

#pragma unroll
    for (int kernel_x = 0; kernel_x < kernel_size; ++kernel_x) {
      int sample_x = outmap_x - kernel_x * dilation + padding;

      const int sample_x_div_stride = sample_x / stride;

      if (sample_x - sample_x_div_stride * stride == 0) {
        sample_x = sample_x_div_stride;

        if ((sample_x >= 0) && (sample_x < sample.x)) {
          int sample_offset = sample_base + sample_x;
          int weight_offset = weight_base + kernel_x;
          for (int d = 0; d < divisor; d++) {
            value += weight_data[weight_offset] * input_data[sample_offset];
            sample_offset += sample_size;
            weight_offset += kernel_size;
          }
        }
      }
    }
    output_data[output_offset] = value;
  }
}

template <typename T, int K>
__global__ void forward_kernel(const T *input_data, T *output_data,
                               const T *weight_data, const T *bias_data,
                               const int output_data_size, const int3 sample,
                               const int3 outmap, const int2 kernel,
                               const int2 stride, const int2 padding,
                               const int2 dilation, const int divisor) {
  // sample/outmap (x, y, z) == (width, height, channels)
  // outmap.z == sample.z / divisor
  // sample/outmap/kernel size == width * height
  const auto &sample_channels = sample.z;
  const auto &outmap_channels = outmap.z;
  const auto sample_size = sample.x * sample.y;
  const auto outmap_size = outmap.x * outmap.y;

  const int KERNEL_X = (K != 0) ? K : kernel.x;
  const int KERNEL_Y = (K != 0) ? K : kernel.y;
  const auto kernel_size = KERNEL_X * KERNEL_Y;

  NBLA_CUDA_KERNEL_LOOP(output_offset, output_data_size) {
    const int s = output_offset / (outmap_channels * outmap_size);
    const int outmap_y = (output_offset / outmap.x) % outmap.y;
    const int outmap_x = output_offset % outmap.x;
    const int outmap_chan = (output_offset / outmap_size) % outmap_channels;
    const int sample_chan = outmap_chan * divisor;
    const int sample_base = (s * sample_channels + sample_chan) * sample_size;
    const int weight_base = sample_chan * kernel_size;
    int weight_incr = 0;

    T value = bias_data ? bias_data[outmap_chan] : (T)0;

#pragma unroll
    for (int k_y = 0; k_y < KERNEL_Y; ++k_y) {
#pragma unroll
      for (int k_x = 0; k_x < KERNEL_X; ++k_x) {
        int sample_y = outmap_y - k_y * dilation.y + padding.y;
        int sample_x = outmap_x - k_x * dilation.x + padding.x;

        const int sample_y_div_stride = sample_y / stride.y;
        const int sample_x_div_stride = sample_x / stride.x;

        if ((sample_y - sample_y_div_stride * stride.y == 0) &&
            (sample_x - sample_x_div_stride * stride.x == 0)) {
          sample_y = sample_y_div_stride;
          sample_x = sample_x_div_stride;

          if ((sample_y >= 0) && (sample_y < sample.y) && //
              (sample_x >= 0) && (sample_x < sample.x)) {
            int weight_offset = weight_base + weight_incr;
            int sample_offset = sample_base + sample_y * sample.x + sample_x;
            for (int d = 0; d < divisor; d++) {
              value += weight_data[weight_offset] * input_data[sample_offset];
              sample_offset += sample_size;
              weight_offset += kernel_size;
            }
          }
        }
        weight_incr++;
      }
    }
    output_data[output_offset] = value;
  }
}

template <typename T, int K>
__global__ void
backprop_input(T *input_grad, const T *output_grad, const T *weight_data,
               const int input_data_size, const int2 sample, const int2 outmap,
               const int kernel, const int stride, const int padding,
               const int dilation, const int divisor) {
  // sample/outmap (x, y) == (length, channels)
  // outmap.y == sample.y / divisor
  const auto &sample_channels = sample.y;
  const auto &outmap_channels = outmap.y;
  const auto &sample_size = sample.x;
  const auto &outmap_size = outmap.x;

  const int kernel_size = (K != 0) ? K : kernel;

  NBLA_CUDA_KERNEL_LOOP(input_offset, input_data_size) {
    const int s = input_offset / (sample_channels * sample_size);
    const int sample_x = input_offset % sample_size;
    const int sample_chan = (input_offset / sample_size) % sample_channels;
    const int outmap_chan = sample_chan / divisor;
    const int outmap_base = (s * outmap_channels + outmap_chan) * outmap_size;
    int weight_offset = sample_chan * kernel;

    T value = 0;

#pragma unroll
    for (int k = 0; k < kernel_size; ++k) {
      const int outmap_x = sample_x * stride + k * dilation - padding;

      if ((outmap_x >= 0) && (outmap_x < outmap_size)) {
        const int output_offset = outmap_base + outmap_x;
        value += weight_data[weight_offset] * output_grad[output_offset];
      }
      weight_offset++;
    }
    input_grad[input_offset] += value;
  }
}

template <typename T, int K>
__global__ void
backprop_input(T *input_grad, const T *output_grad, const T *weight_data,
               const int input_data_size, const int3 sample, const int3 outmap,
               const int2 kernel, const int2 stride, const int2 padding,
               const int2 dilation, const int divisor) {
  // sample/outmap (x, y, z) == (width, height, channels)
  // outmap.z == sample.z / divisor
  // sample/outmap/kernel size == width * height
  const auto &sample_channels = sample.z;
  const auto &outmap_channels = outmap.z;

  const int sample_size = sample.x * sample.y;
  const int outmap_size = outmap.x * outmap.y;

  const int KERNEL_X = (K != 0) ? K : kernel.x;
  const int KERNEL_Y = (K != 0) ? K : kernel.y;
  const int kernel_size = KERNEL_X * KERNEL_Y;

  NBLA_CUDA_KERNEL_LOOP(input_offset, input_data_size) {
    const int s = input_offset / (sample_channels * sample_size);
    const int sample_y = (input_offset / sample.x) % sample.y;
    const int sample_x = input_offset % sample.x;
    const int sample_chan = (input_offset / sample_size) % sample_channels;
    const int outmap_chan = sample_chan / divisor;
    const int outmap_base = (s * outmap_channels + outmap_chan) * outmap_size;
    int weight_offset = sample_chan * kernel_size;

    T value = 0;

#pragma unroll
    for (int k_y = 0; k_y < KERNEL_Y; ++k_y) {
#pragma unroll
      for (int k_x = 0; k_x < KERNEL_X; ++k_x) {
        const int outmap_y = sample_y * stride.y + k_y * dilation.y - padding.y;
        const int outmap_x = sample_x * stride.x + k_x * dilation.x - padding.x;

        if ((outmap_y >= 0) && (outmap_y < outmap.y) && //
            (outmap_x >= 0) && (outmap_x < outmap.x)) {
          int output_offset = outmap_base + outmap_y * outmap.x + outmap_x;
          value += weight_data[weight_offset] * output_grad[output_offset];
        }
        weight_offset++;
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
                 const int padding, const int dilation, const int divisor) {
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

  const int sample_chan = chan;
  const int outmap_chan = chan / divisor;
  const int sample_stride = sample_channels * sample_size;
  const int outmap_stride = outmap_channels * outmap_size;
  const int sample_offset = sample_chan * sample_size;
  const int outmap_offset = outmap_chan * outmap_size;
  const int outmap_skip_x = k_x * dilation;

  T weight_grad_value = 0; // thread local weight gradient
  T bias_grad_value = 0;   // thread local bias gradient

  // Each thread block (of sample-channels * kernel-size blocks)
  // computes a weight gradient. It also computes a bias gradient
  // even if that is done kernel-size times more often than
  // needed (but we won't gain with a conditional).
  for (int s = samp; s < batch_size; s += warps) {
    const int sample_base = s * sample_stride + sample_offset;
    const int outmap_base = s * outmap_stride + outmap_offset;

    for (int idx = lane; idx < outmap.x; idx += warpSize) {
      const T outmap_grad = output_grad[outmap_base + idx];
      int sample_x = idx - outmap_skip_x + padding;
      const int sample_x_div_stride = sample_x / stride;

      if (sample_x - sample_x_div_stride * stride == 0) {
        sample_x = sample_x_div_stride;

        if ((sample_x >= 0) && (sample_x < sample.x)) {
          const T sample_data = input_data[sample_base + sample_x];
          weight_grad_value += sample_data * outmap_grad;
        }
      }
      bias_grad_value += outmap_grad;
    }
  }
  __syncthreads();

  // Accumulate thread local gradients.
  weight_grad_value = blockReduceSum(weight_grad_value);
  bias_grad_value = blockReduceSum(bias_grad_value);

  // The first thread has the accumulated gradients. The bias
  // gradient is needed only once per output channel.
  if (threadIdx.x == 0) {
    const int weight_offset = k_x + kernel * sample_chan;
    weight_grad[weight_offset] += weight_grad_value;
    if (bias_grad && (blockIdx.x % (kernel * divisor) == 0)) {
      bias_grad[outmap_chan] += bias_grad_value;
    }
  }
}

template <typename T>
__global__ void
backprop_weights(const T *output_grad, const T *input_data, T *weight_grad,
                 T *bias_grad, const int batch_size, const int3 sample,
                 const int3 outmap, const int2 kernel, const int2 stride,
                 const int2 padding, const int2 dilation, const int divisor) {
  const auto &sample_channels = sample.z;
  const auto &outmap_channels = outmap.z;

  // gridDim.x == blocks == outmap_channels * kernel.y * kernel.x == weights
  // blockDim.x == threads == min(batch_size * warpSize, 1024)
  // each block computes one weight value

  const int k_x = blockIdx.x % kernel.x;
  const int k_y = (blockIdx.x / kernel.x) % kernel.y;
  const int chan = blockIdx.x / (kernel.y * kernel.x);
  const int samp = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int warps = blockDim.x / warpSize;

  const int sample_chan = chan;
  const int outmap_chan = chan / divisor;
  const int kernel_size = kernel.x * kernel.y;
  const int sample_stride = sample_channels * sample.y * sample.x;
  const int outmap_stride = outmap_channels * outmap.y * outmap.x;
  const int sample_offset = sample_chan * sample.y * sample.x;
  const int outmap_offset = outmap_chan * outmap.y * outmap.x;
  const int outmap_skip_x = k_x * dilation.x;
  const int outmap_skip_y = k_y * dilation.y;

  T weight_grad_value = 0; // thread local weight gradient
  T bias_grad_value = 0;   // thread local bias gradient

  // Each thread block (of sample-channels * kernel-size blocks)
  // computes a weight gradient. It also computes a bias gradient
  // even if that is done kernel-size times more often than
  // needed (but we won't gain with a conditional).
  for (int s = samp; s < batch_size; s += warps) {
    const int sample_base = s * sample_stride + sample_offset;
    const int outmap_base = s * outmap_stride + outmap_offset;

    for (int idx = lane; idx < outmap.x * outmap.y; idx += warpSize) {
      const T outmap_grad = output_grad[outmap_base + idx];

      int sample_x = (idx % outmap.x) - outmap_skip_x + padding.x;
      int sample_y = (idx / outmap.x) - outmap_skip_y + padding.y;

      const int sample_x_div_stride = sample_x / stride.x;
      const int sample_y_div_stride = sample_y / stride.y;

      if ((sample_x - sample_x_div_stride * stride.x == 0) &&
          (sample_y - sample_y_div_stride * stride.y == 0)) {
        sample_x = sample_x_div_stride;
        sample_y = sample_y_div_stride;

        if ((sample_x >= 0) && (sample_x < sample.x) && //
            (sample_y >= 0) && (sample_y < sample.y)) {
          const int sample_offset = (sample_y * sample.x) + sample_x;
          const T sample_data = input_data[sample_base + sample_offset];
          weight_grad_value += sample_data * outmap_grad;
        }
      }
      bias_grad_value += outmap_grad;
    }
  }
  __syncthreads();

  // Accumulate thread local gradients.
  weight_grad_value = blockReduceSum(weight_grad_value);
  bias_grad_value = blockReduceSum(bias_grad_value);

  // The first thread has the accumulated gradients. The bias
  // gradient is needed only once per output channel.
  if (threadIdx.x == 0) {
    const int weight_offset = sample_chan * kernel_size + k_y * kernel.x + k_x;
    weight_grad[weight_offset] += weight_grad_value;
    if (bias_grad && (blockIdx.x % (kernel_size * divisor) == 0)) {
      bias_grad[outmap_chan] += bias_grad_value;
    }
  }
}

} // namespace depthwise_deconvolution_cuda

using namespace depthwise_deconvolution_cuda;

template <typename T>
void DepthwiseDeconvolutionCuda<T>::setup_impl(const Variables &inputs,
                                               const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));
  DepthwiseDeconvolution<T>::setup_impl(inputs, outputs);

  Variable *const input = inputs[0];
  Variable *const output = outputs[0];

  input_data_size_ = input->size();
  output_data_size_ = output->size();

  NBLA_CHECK(inputs[1]->size() <= 65536, error_code::unclassified,
             "GPU implementation limit reached: output-channels x filter-size "
             "can not be more than 65536."); // see implementation note

  if (this->kernel_shape_.size() == 1) {
    sample_1d_ = make_int2(this->sample_shape_[0], this->sample_channels_);
    outmap_1d_ = make_int2(this->outmap_shape_[0], this->outmap_channels_);
    kernel_1d_ = this->kernel_shape_[0];
    stride_1d_ = this->stride_[0];
    padding_1d_ = this->padding_[0];
    dilation_1d_ = this->dilation_[0];
  } else {
    sample_2d_ = make_int3(this->sample_shape_[1], this->sample_shape_[0],
                           this->sample_channels_);
    outmap_2d_ = make_int3(this->outmap_shape_[1], this->outmap_shape_[0],
                           this->outmap_channels_);
    kernel_2d_ = make_int2(this->kernel_shape_[1], this->kernel_shape_[0]);
    stride_2d_ = make_int2(this->stride_[1], this->stride_[0]);
    padding_2d_ = make_int2(this->padding_[1], this->padding_[0]);
    dilation_2d_ = make_int2(this->dilation_[1], this->dilation_[0]);
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, std::stoi(this->ctx_.device_id));
  max_threads_per_block_ = prop.maxThreadsPerBlock;
  warp_size_ = prop.warpSize;
}

template <typename T>
void DepthwiseDeconvolutionCuda<T>::forward_impl(const Variables &inputs,
                                                 const Variables &outputs) {
  cuda_set_device(std::stoi(this->ctx_.device_id));

  Variable *const input = inputs[0];
  Variable *const weight = inputs[1];
  Variable *const bias = (inputs.size() == 3) ? inputs[2] : nullptr;
  Variable *const output = outputs[0];

  const Tc *input_data = input->get_data_pointer<Tc>(this->ctx_);
  const Tc *weight_data = weight->get_data_pointer<Tc>(this->ctx_);
  const Tc *bias_data =
      (bias) ? bias->get_data_pointer<Tc>(this->ctx_) : nullptr;
  Tc *output_data = output->cast_data_and_get_pointer<Tc>(this->ctx_);

  const int threads = max_threads_per_block_;
  const int blocks = (output_data_size_ + threads - 1) / threads;

  if (this->kernel_shape_.size() == 1) {
    if (kernel_1d_ == 3) {
      forward_kernel<Tc, 3><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_1d_, outmap_1d_, kernel_1d_, stride_1d_, padding_1d_,
          dilation_1d_, this->divisor_);
    } else if (kernel_1d_ == 5) {
      forward_kernel<Tc, 5><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_1d_, outmap_1d_, kernel_1d_, stride_1d_, padding_1d_,
          dilation_1d_, this->divisor_);
    } else {
      forward_kernel<Tc, 0><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_1d_, outmap_1d_, kernel_1d_, stride_1d_, padding_1d_,
          dilation_1d_, this->divisor_);
    }
  } else {
    if ((kernel_2d_.x == 3) && (kernel_2d_.y == 3)) {
      forward_kernel<Tc, 3><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_2d_, outmap_2d_, kernel_2d_, stride_2d_, padding_2d_,
          dilation_2d_, this->divisor_);
    } else if ((kernel_2d_.x == 5) && (kernel_2d_.y == 5)) {
      forward_kernel<Tc, 5><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_2d_, outmap_2d_, kernel_2d_, stride_2d_, padding_2d_,
          dilation_2d_, this->divisor_);
    } else {
      forward_kernel<Tc, 0><<<blocks, threads>>>(
          input_data, output_data, weight_data, bias_data, output_data_size_,
          sample_2d_, outmap_2d_, kernel_2d_, stride_2d_, padding_2d_,
          dilation_2d_, this->divisor_);
    }
  }
}

template <typename T>
void DepthwiseDeconvolutionCuda<T>::backward_impl(
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

  const Tc *input_data = input->get_data_pointer<Tc>(this->ctx_);
  const Tc *weight_data = weight->get_data_pointer<Tc>(this->ctx_);
  const Tc *output_grad = output->get_grad_pointer<Tc>(this->ctx_);

  Tc *input_grad = nullptr;
  if (propagate_down[0]) {
    if (!accum[0])
      input->grad()->zero(); // before cast!
    input_grad = input->cast_grad_and_get_pointer<Tc>(this->ctx_);
  }

  Tc *weight_grad = nullptr;
  if (propagate_down[1]) {
    if (!accum[1])
      weight->grad()->zero(); // before cast!
    weight_grad = weight->cast_grad_and_get_pointer<Tc>(this->ctx_);
  }

  Tc *bias_grad = nullptr;
  if (inputs.size() == 3 && propagate_down[2]) {
    if (!accum[2])
      bias->grad()->zero(); // before cast!
    bias_grad = bias->cast_grad_and_get_pointer<Tc>(this->ctx_);
  }

  if (input_grad) {
    // Compute the input gradient. Each thread of threads times
    // blocks computes one input gradient value.
    const int threads = max_threads_per_block_;
    const int blocks = (input_data_size_ + threads - 1) / threads;
    if (this->kernel_shape_.size() == 1) {
      if (kernel_1d_ == 3) {
        backprop_input<Tc, 3><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_1d_,
            outmap_1d_, kernel_1d_, stride_1d_, padding_1d_, dilation_1d_,
            this->divisor_);
      } else if (kernel_1d_ == 5) {
        backprop_input<Tc, 5><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_1d_,
            outmap_1d_, kernel_1d_, stride_1d_, padding_1d_, dilation_1d_,
            this->divisor_);
      } else {
        backprop_input<Tc, 0><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_1d_,
            outmap_1d_, kernel_1d_, stride_1d_, padding_1d_, dilation_1d_,
            this->divisor_);
      }
    } else {
      if ((kernel_2d_.x == 3) && (kernel_2d_.y == 3)) {
        backprop_input<Tc, 3><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_2d_,
            outmap_2d_, kernel_2d_, stride_2d_, padding_2d_, dilation_2d_,
            this->divisor_);
      } else if ((kernel_2d_.x == 5) && (kernel_2d_.y == 5)) {
        backprop_input<Tc, 5><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_2d_,
            outmap_2d_, kernel_2d_, stride_2d_, padding_2d_, dilation_2d_,
            this->divisor_);
      } else {
        backprop_input<Tc, 0><<<blocks, threads>>>(
            input_grad, output_grad, weight_data, input_data_size_, sample_2d_,
            outmap_2d_, kernel_2d_, stride_2d_, padding_2d_, dilation_2d_,
            this->divisor_);
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
    const int batch_size = this->batch_size_;
    const int threads = min(batch_size * warp_size_, max_threads_per_block_);
    if (this->kernel_shape_.size() == 1) {
      const int kernel_size = kernel_1d_;
      const int sample_channels = sample_1d_.y;
      const int blocks = sample_channels * kernel_size;
      backprop_weights<Tc><<<blocks, threads>>>(
          output_grad, input_data, weight_grad, bias_grad, batch_size,
          sample_1d_, outmap_1d_, kernel_1d_, stride_1d_, padding_1d_,
          dilation_1d_, this->divisor_);
    } else {
      const int kernel_size = kernel_2d_.x * kernel_2d_.y;
      const int sample_channels = sample_2d_.z;
      const int blocks = sample_channels * kernel_size;
      backprop_weights<Tc><<<blocks, threads>>>(
          output_grad, input_data, weight_grad, bias_grad, batch_size,
          sample_2d_, outmap_2d_, kernel_2d_, stride_2d_, padding_2d_,
          dilation_2d_, this->divisor_);
    }
    NBLA_CUDA_KERNEL_CHECK();
  }

  if (bias_grad && !weight_grad) {
    // Normally the bias gradient is computed along with the weight
    // gradient in the same cuda kernel. In the unlikely event that
    // only the bias gradient is requested, we fall back to this
    // slower method.
    if (this->kernel_shape_.size() == 1) {
      const int outmap_size = outmap_1d_.x;
      const int outmap_channels = outmap_1d_.y;
      const Tc *ones =
          static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
              outmap_size, get_dtype<Tc>(), this->ctx_));

      for (int samp = 0; samp < this->batch_size_; ++samp) {
        const int output_offset = samp * outmap_channels * outmap_size;
        cuda_gemv<Tc>(device_, bias_grad, &output_grad[output_offset],
                      outmap_size, outmap_channels, true, ones, outmap_size, 1,
                      1);
      }
    } else {
      const int outmap_size = outmap_2d_.x * outmap_2d_.y;
      const int outmap_channels = outmap_2d_.z;
      const Tc *ones =
          static_cast<const Tc *>(SingletonManager::get<NNabla>()->ones(
              outmap_size, get_dtype<Tc>(), this->ctx_));

      for (int samp = 0; samp < this->batch_size_; ++samp) {
        const int output_offset = samp * outmap_channels * outmap_size;
        cuda_gemv<Tc>(device_, bias_grad, &output_grad[output_offset],
                      outmap_size, outmap_channels, true, ones, outmap_size, 1,
                      1);
      }
    }
  }
}

} // namespace nbla
