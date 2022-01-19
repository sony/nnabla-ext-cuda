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

#ifndef __NBLA_CUDA_UTILS_STFT_HPP__
#define __NBLA_CUDA_UTILS_STFT_HPP__

#include <string>

namespace stft {
enum WINDOW_TYPE {
  hanning = 0,
  hamming,
  rectangular,
};
}

static stft::WINDOW_TYPE string_to_window_type(const std::string &type) {
  using WINDOW_TYPE = stft::WINDOW_TYPE;

  if (type == "hanning") {
    return WINDOW_TYPE::hanning;
  } else if (type == "hamming") {
    return WINDOW_TYPE::hamming;
  }
  // type == "rectangular"
  return WINDOW_TYPE::rectangular;
}

#endif
