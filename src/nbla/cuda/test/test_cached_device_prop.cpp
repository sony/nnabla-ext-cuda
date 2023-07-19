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
#include "gtest/gtest.h"
#include <chrono>

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/cuda.hpp>
#include <nbla/cuda/cudnn/cudnn.hpp>

namespace nbla {

TEST(get_device_properties, TestIfGetProp) {
  std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
  shared_ptr<cudaDeviceProp> device_prop =
      SingletonManager::get<Cuda>()->get_device_properties();
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  auto dur1 =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  const auto min_blocks_per_sm = device_prop->maxThreadsPerMultiProcessor;
  auto min_blocks = min_blocks_per_sm * device_prop->multiProcessorCount;
  ASSERT_GT(min_blocks, 0);

  t0 = std::chrono::steady_clock::now();
  device_prop = SingletonManager::get<Cuda>()->get_device_properties();
  t1 = std::chrono::steady_clock::now();
  auto dur2 =
      std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

  ASSERT_GT(dur1, dur2);
  ASSERT_EQ(min_blocks_per_sm, device_prop->maxThreadsPerMultiProcessor);
  printf("assert {%d} > {%d} is true!\n", dur1, dur2);
}
} // namespace nbla
