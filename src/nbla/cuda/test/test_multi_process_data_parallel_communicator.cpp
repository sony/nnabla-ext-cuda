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
#ifdef FEATURE_DIST_TRAIN
#include "gtest/gtest.h"

#include <nbla/computation_graph/computation_graph.hpp>
#include <nbla/computation_graph/variable.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/communicator/multi_process_data_parallel_communicator.hpp>
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/function/callback.hpp>

#include <array>

namespace nbla {

extern void increment_vector(cudaStream_t stream, float *vec, size_t size);

template <typename T, typename... Ts>
std::unique_ptr<T> make_unique(Ts &&... params) {
  return std::unique_ptr<T>(new T(std::forward<Ts>(params)...));
}

auto ignore1 = [](void *obj, const Variables &, const Variables &) {};
auto ignore2 = [](void *obj, const Variables &inputs, const Variables &outputs,
                  const vector<bool> &propagate_down,
                  const vector<bool> &accum) {};
auto ignore3 = [](void *obj) {};

struct MockCommunicator
    : public MultiProcessDataParallelCommunicatorNccl<float> {
  using MultiProcessDataParallelCommunicatorNccl<
      float>::MultiProcessDataParallelCommunicatorNccl;
  virtual void all_reduce(float *buffer, size_t n_param, cudaStream_t stream,
                          bool division, bool inplace, const string &group) {
    if (this->pipeline) {
      this->pipeline(buffer, n_param, stream);
    }
  }
  std::function<void(float *, size_t, cudaStream_t)> pipeline;
};

Context ctx;
std::unique_ptr<MockCommunicator> comm = nullptr;
class MultiProcessDataParallelCommunicatorEnvironment
    : public ::testing::Environment {
protected:
  virtual void SetUp() {
    if (comm == nullptr) {
      init_cudnn();
      ctx = Context{{"cudnn|float", "cuda:float"}, "CudaCachedArray", "0"};
      comm = make_unique<MockCommunicator>(ctx);
      comm->init();
    }
  }
  virtual void TearDown() { comm = nullptr; }
};
auto const communicator_env = ::testing::AddGlobalTestEnvironment(
    new MultiProcessDataParallelCommunicatorEnvironment);

TEST(MultiProcessDataParallelCommunicatorTest, PackingAndUnpacking) {
  // Create network
  auto a = std::make_shared<CgFunction>(std::make_shared<Callback>(
      ctx, nullptr, ignore1, ignore1, ignore2, ignore3));
  auto b = std::make_shared<CgFunction>(std::make_shared<Callback>(
      ctx, nullptr, ignore1, ignore1, ignore2, ignore3));
  auto c = std::make_shared<CgFunction>(std::make_shared<Callback>(
      ctx, nullptr, ignore1, ignore1, ignore2, ignore3));
  auto d = std::make_shared<CgFunction>(std::make_shared<Callback>(
      ctx, nullptr, ignore1, ignore1, ignore2, ignore3));
  auto v1 = std::make_shared<CgVariable>(Shape_t{2, 1, 1}, true);
  auto v2 = std::make_shared<CgVariable>(Shape_t{1, 1, 1}, true);
  auto v3 = std::make_shared<CgVariable>(Shape_t{1, 1, 1}, true);
  auto v4 = std::make_shared<CgVariable>(Shape_t{1, 1, 1}, true);
  auto h1 = connect(a, {v1}, 1);
  h1.push_back(v2);
  auto h2 = connect(b, h1, 1);
  h2.push_back(v3);
  auto h3 = connect(c, h2, 1);
  h3.push_back(v4);
  auto h4 = connect(d, h3, 1);

  // Set data
  std::vector<float> data;
  data = {20.0f, 10.0f};
  cudaMemcpy(v1->variable()->cast_grad_and_get_pointer<float>(ctx), data.data(),
             data.size() * sizeof(float), cudaMemcpyHostToDevice);
  data = {30.0f};
  cudaMemcpy(v2->variable()->cast_grad_and_get_pointer<float>(ctx), data.data(),
             data.size() * sizeof(float), cudaMemcpyHostToDevice);
  data = {40.0f};
  cudaMemcpy(v3->variable()->cast_grad_and_get_pointer<float>(ctx), data.data(),
             data.size() * sizeof(float), cudaMemcpyHostToDevice);
  data = {50.0f};
  cudaMemcpy(v4->variable()->cast_grad_and_get_pointer<float>(ctx), data.data(),
             data.size() * sizeof(float), cudaMemcpyHostToDevice);

  // Setup communicator
  int call_pipeline_count = 0;
  std::vector<size_t> n_params;
  std::vector<std::vector<float>> buffers;
  comm->pipeline = [&](float *buffer, size_t n_param, cudaStream_t stream) {
    call_pipeline_count += 1;
    n_params.push_back(n_param);
    buffers.emplace_back(n_param);
    cudaMemcpyAsync(buffers.back().data(), buffer, sizeof(float) * n_param,
                    cudaMemcpyDeviceToHost, stream);

    // increment all data
    increment_vector(stream, buffer, n_param);
  };

  auto process = comm->all_reduce_callback(
      {v1->variable()->grad(), v2->variable()->grad(), v3->variable()->grad(),
       v4->variable()->grad()},
      2);

  process->on_finish_function_backward(d);
  process->on_finish_function_backward(c);
  process->on_finish_function_backward(b);
  process->on_finish_function_backward(a);
  process->on_finish_backward();

  NBLA_CUDA_CHECK(cudaStreamSynchronize(0));
  EXPECT_EQ(3, call_pipeline_count);
  EXPECT_EQ(std::vector<size_t>({2, 2, 1}), n_params);
  EXPECT_EQ(3, buffers.size());
  for (auto i = 0; i < buffers.size(); ++i) {
    if (i == 0) {
      EXPECT_EQ(2, buffers[i].size());
      EXPECT_FLOAT_EQ(50.0f, buffers[i][0]);
      EXPECT_FLOAT_EQ(40.0f, buffers[i][1]);
    } else if (i == 1) {
      EXPECT_EQ(2, buffers[i].size());
      EXPECT_FLOAT_EQ(30.0f, buffers[i][0]);
      EXPECT_FLOAT_EQ(20.0f, buffers[i][1]);
    } else {
      EXPECT_EQ(1, buffers[i].size());
      EXPECT_FLOAT_EQ(10.0f, buffers[i][0]);
    }
  }

  data.resize(2);
  cudaMemcpy(data.data(), v1->variable()->cast_grad_and_get_pointer<float>(ctx),
             2 * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(21.0f, data[0]);
  EXPECT_FLOAT_EQ(11.0f, data[1]);
  cudaMemcpy(data.data(), v2->variable()->cast_grad_and_get_pointer<float>(ctx),
             1 * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(31.0f, data[0]);
  cudaMemcpy(data.data(), v3->variable()->cast_grad_and_get_pointer<float>(ctx),
             1 * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(41.0f, data[0]);
  cudaMemcpy(data.data(), v4->variable()->cast_grad_and_get_pointer<float>(ctx),
             1 * sizeof(float), cudaMemcpyDeviceToHost);
  EXPECT_FLOAT_EQ(51.0f, data[0]);
}
}
#endif