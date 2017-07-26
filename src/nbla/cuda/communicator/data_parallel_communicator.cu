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

#include <nbla/cuda/communicator/data_parallel_communicator.hpp>
#include <nbla/cuda/common.hpp>

#include <algorithm>
#include <memory>
#include <cstdlib>

namespace nbla {

using std::make_shared;


template<typename T>
__global__ void kernel_divide_inplace(const int size, const int n_devices,
    T *dw) {
  NBLA_CUDA_KERNEL_LOOP(i, size) {
    dw[i] /= n_devices;
  }
}


template<typename T>
DataParallelCommunicatorNccl<T>::DataParallelCommunicatorNccl(const Context &ctx) : DataParallelCommunicator<T>(ctx) {}

template<typename T>
DataParallelCommunicatorNccl<T>::~DataParallelCommunicatorNccl() {
  if (this->initialized_) {
    for (int i = 0; i < device_ids_.size(); ++i) {
      ncclCommDestroy(comms_[i]);
      NBLA_CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
  }
}

template<typename T>
void DataParallelCommunicatorNccl<T>::init() {
  Communicator::init();
  try {
    // Set gpu information
    for (auto ctx: this->contexts_) {
      this->device_ids_.push_back(std::stoi(ctx.device_id));
    }
    this->n_devices_ = this->device_ids_.size();

    // Initialize stream and communicator
    for (int i = 0; i < n_devices_; ++i) {
      cuda_set_device(device_ids_[i]);
      // Stream
      cudaStream_t stream;
      NBLA_CUDA_CHECK(cudaStreamCreate(&stream));
      streams_.push_back(stream);

      // NCCL Comm
      ncclComm_t comm;
      comms_.push_back(comm);
    }
    ncclResult_t res = ncclCommInitAll(comms_.data(), this->n_devices_, this->device_ids_.data());
    if (res != 0) {
      NBLA_ERROR(error_code::target_specific,
          "ncclCommInitAll fails with %d");
    }
  } catch (...) {
    this->initialized_ = false;
  }
  this->initialized_ = true;
}

template<typename T>
void DataParallelCommunicatorNccl<T>::reduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU reduce is not implemented.")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::allreduce(bool division) {
  // Sync all devices
  wait_by_devices_synchronization();

  // Once sync to prevent the hang where the memcpy occurs during the allreduce.
  this->sync_all_params();

  // Inpalce allreduce
  for (int i = 0; i < device_ids_.size(); ++i) {  // device-loop
    Context ctx = this->contexts_[i];
    auto device_id = device_ids_[i];
    cuda_set_device(device_id);

    auto func_named_param = this->device_func_named_param_[i];
    auto comm = comms_[i];
    auto stream = streams_[i];
    auto size = func_named_param.size();

    for (auto elm : func_named_param) {          // function-loop
      VariablePtr vp = elm.second;
      auto n_param = vp->size();

      const T *dw0 = vp->get_grad_pointer<T>(ctx);
      T *dw1 = vp->cast_grad_and_get_pointer<T>(ctx);
      ncclResult_t res = ncclAllReduce(
          dw0, dw1,
          n_param, ncclFloat, ncclSum, //TODO: address ncclFloat
          comm,
          stream);
      if (res != 0) {
        NBLA_ERROR(error_code::target_specific,
            "ncclAllReduce fails with %d.", res);
      }
    }
  }
  // Divide using the same streams
  divide_by_num_divices(division);

  // Sync streams
  wait_by_streams_synchronization();
}

template<typename T>
void DataParallelCommunicatorNccl<T>::reducescatter(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU reducescatter is not implemented.")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::bcast() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU bcast is not implemented.")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::allgather() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU allgather is not implemented.")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU reduce_async is not implemented.")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::allreduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU allreduce_async is not implemented.")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::reducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU reducescatter_async is not implemented.")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::bcast_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU bcast_async is not implemented.")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU allgather_async is not implemented.")
}

template<typename T>
vector<string> DataParallelCommunicatorNccl<T>::allowed_array_classes() {
  NBLA_ERROR(error_code::not_implemented,
      "Derived class of DataParallelCommunicatorNccl must implement allowed_array_classes().")
}

template<typename T>
void DataParallelCommunicatorNccl<T>::wait_by_devices_synchronization() {
  for (int i = 0; i < device_ids_.size(); ++i) {
    cuda_device_synchronize(device_ids_[i]);
  }
}

template<typename T>
void DataParallelCommunicatorNccl<T>::wait_by_streams_synchronization() {
  for (int i = 0; i < device_ids_.size(); ++i) {
    cuda_set_device(device_ids_[i]);
    NBLA_CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
  }
}

template<typename T>
void DataParallelCommunicatorNccl<T>::divide_by_num_divices(bool division) {
  if (division) {
    for (int i = 0; i < device_ids_.size(); ++i) {
      auto device_id = device_ids_[i];
      cuda_set_device(device_id);

      Context ctx = this->contexts_[i];
      auto func_named_param = this->device_func_named_param_[i];
      auto stream = streams_[i];
      for (auto elm : func_named_param) {
        VariablePtr vp = elm.second;
        T *dw = vp->cast_grad_and_get_pointer<T>(ctx);
        auto n_param = vp->size();
        NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(
            kernel_divide_inplace, stream, n_param, n_devices_, dw);
      }
    }
  }
}

template<typename T>
void DataParallelCommunicatorNccl<T>::sync_all_params() {
 for (int i = 0; i < device_ids_.size(); ++i) {  // device-loop
    Context ctx = this->contexts_[i];
    auto device_id = device_ids_[i];

    auto func_named_param = this->device_func_named_param_[i];
    auto size = func_named_param.size();

    for (auto elm : func_named_param) {          // function-loop
      VariablePtr vp = elm.second;

      // If the arrays are different, output the warning.
      this->check_array_class(ctx, vp);

      // Sync
      vp->get_grad_pointer<T>(ctx);
    }
  }
}

template class DataParallelCommunicatorNccl<float>;
}
