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

/** ImageAugmentation
 */
#ifndef __NBLA_CUDA_FUNCTION_IMAGEAUGMENTATION_HPP__
#define __NBLA_CUDA_FUNCTION_IMAGEAUGMENTATION_HPP__

#include <nbla/cuda/cuda.hpp>
#include <nbla/function/image_augmentation.hpp>
namespace nbla {
/** @copydoc ImageAugmentation
*/

template <typename T>
class ImageAugmentationCuda : public ImageAugmentation<T> {

protected:
  Variable curand_state_;

public:
  typedef typename CudaType<T>::type Tc;
  explicit ImageAugmentationCuda(const Context &ctx, const vector<int> &shape,
                                 const vector<int> &pad, float min_scale,
                                 float max_scale, float angle,
                                 float aspect_ratio, float distortion,
                                 bool flip_lr, bool flip_ud, float brightness,
                                 bool brightness_each, float contrast,
                                 float contrast_center, bool contrast_each,
                                 float noise, int seed)
      : ImageAugmentation<T>(ctx, shape, pad, min_scale, max_scale, angle,
                             aspect_ratio, distortion, flip_lr, flip_ud,
                             brightness, brightness_each, contrast,
                             contrast_center, contrast_each, noise, seed),
        device_(std::stoi(ctx.device_id)) {}
  virtual ~ImageAugmentationCuda() {}
  virtual string name() { return "ImageAugmentationCuda"; }
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
};
}

#endif
