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
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/function/pad.hpp>
#include <nbla/variable.hpp>

// TODO: Remove these #includes. Only for debug.
#include <iostream>
#include <typeinfo>


namespace nbla {

template <typename T>
__global__ void kernel_pad_forward( const int num,const T *x, T *y, const float value,
                                    int pad_top, int pad_left,
                                    int input_height, int input_width,
                                    int output_height, int output_width) {
	NBLA_CUDA_KERNEL_LOOP(i, num)
	{
		int num_channel = i / output_width;
		const int pad_h = num_channel % output_height;
		const int pad_w = i % output_width;
		const int height = pad_h - pad_top;
		const int width = pad_w - pad_left;
		num_channel /= output_height;

		if((height >= 0 && height < input_height) &&
		   (width  >= 0 && width  < input_width)) {
			y[i] = x[(num_channel * input_height + height) * input_width + width];
		}else {
			y[i] = (T)value;
		}
	}

}

template <typename T, bool accum>
__global__ void kernel_pad_backward( const int num, T *dx, const T *dy, const float value,
                                    int pad_top, int pad_left,
                                    int input_height, int input_width,
                                    int output_height, int output_width) {

	NBLA_CUDA_KERNEL_LOOP(i, num)
	{
		int num_channel = i / output_width;
		const int pad_h = num_channel % output_height;
		const int pad_w = i % output_width;
		const int height = pad_h - pad_top;
		const int width = pad_w - pad_left;
		num_channel /= output_height;

		if((height >= 0 && height < input_height) &&
		   (width  >= 0 && width  < input_width)) {
			if(accum){
				dx[(num_channel * input_height + height) * input_width + width] += dy[i];
			}
			else{
				dx[(num_channel * input_height + height) * input_width + width] = dy[i];
			}
		}
	}
}

template <typename T>
void PadCuda<T>::setup_impl(const Variables &inputs,
                               const Variables &outputs) {

  Pad<T>::setup_impl(inputs, outputs);
  cuda_set_device(this->device_);

}

template <typename T>
void PadCuda<T>::forward_impl(const Variables &inputs,
                               const Variables &outputs) {

  cuda_set_device(this->device_);

  const Tcu* x = inputs[0]->get_data_pointer<Tcu>(this->ctx_);
  //Tcu *y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_);
  Tcu* y = outputs[0]->cast_data_and_get_pointer<Tcu>(this->ctx_, true);

  vector<int> padding = this->pad_width_;

  // If 1D padding insert top and bottom padding as '0' and use 2D padding code
  if(padding.size() <= 2)
  {
     padding.insert(padding.begin(),0);
     padding.insert(padding.begin(),0);
  }

  int pad_top=padding[0],pad_left=padding[2];
  int input_height=inputs[0]->shape()[2];
  int input_width=inputs[0]->shape()[3];
  int output_height=outputs[0]->shape()[2];
  int output_width=outputs[0]->shape()[3];
  size_t size = outputs[0]->size();

  NBLA_CUDA_LAUNCH_KERNEL_SIMPLE(kernel_pad_forward, size ,x, y,this->constant_value_,
									   padding[0],padding[2],
                                       inputs[0]->shape()[2],inputs[0]->shape()[3],
                                       outputs[0]->shape()[2],outputs[0]->shape()[3]);

}


template <typename T>
void PadCuda<T>::backward_impl(const Variables &inputs,
                                const Variables &outputs,
                                const vector<bool> &propagate_down,
                                const vector<bool> &accum) {

  if (!(propagate_down[0])) {
    return;
  }
  cuda_set_device(this->device_);
  Tcu* g_x{nullptr};
  g_x = inputs[0]->cast_grad_and_get_pointer<Tcu>(this->ctx_, !accum[0]);
  const Tcu* g_y = outputs[0]->get_grad_pointer<Tcu>(this->ctx_);

  vector<int> padding = this->pad_width_;

  // If 1D padding insert top and bottom padding as '0' and use 2D padding code
  if(padding.size() <= 2)
  {
     padding.insert(padding.begin(),0);
     padding.insert(padding.begin(),0);
  }

  int pad_top=padding[0],pad_left=padding[2];
  int input_height=inputs[0]->shape()[2];
  int input_width=inputs[0]->shape()[3];
  int output_height=outputs[0]->shape()[2];
  int output_width=outputs[0]->shape()[3];
  size_t size = outputs[0]->size();

  if (accum[0]) {
	NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_pad_backward<Tcu, true>), size ,g_x, g_y,this->constant_value_,
									   padding[0],padding[2],
                                       inputs[0]->shape()[2],inputs[0]->shape()[3],
                                       outputs[0]->shape()[2],outputs[0]->shape()[3])
  }else {
	NBLA_CUDA_LAUNCH_KERNEL_SIMPLE((kernel_pad_backward<Tcu, false>), size ,g_x, g_y,this->constant_value_,
									   padding[0],padding[2],
                                       inputs[0]->shape()[2],inputs[0]->shape()[3],
                                       outputs[0]->shape()[2],outputs[0]->shape()[3])
  }
}
}