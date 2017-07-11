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

#include <nbla/cuda/communicator/multi_process_data_parallel_communicator.hpp>
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
MultiProcessDataParallelCommunicatorNccl<T>::MultiProcessDataParallelCommunicatorNccl(const Context &ctx) : MultiProcessDataParallelCommunicator<T>(ctx) {}

template<typename T>
MultiProcessDataParallelCommunicatorNccl<T>::~MultiProcessDataParallelCommunicatorNccl() {
  if (this->initialized_) {
    for (int i = 0; i < device_ids_.size(); ++i) {
      ncclCommDestroy(comm_ptr_[i]);
      cudaStreamDestroy(stream_ptr_[i]);
    }
    free(comm_ptr_);
    free(stream_ptr_);
  }
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::init() {
  Communicator::init();
  try {
    // Set gpu information
    for (auto ctx: this->contexts_) {
      this->device_ids_.push_back(std::stoi(ctx.device_id));
    }
    this->n_devices_ = this->device_ids_.size();

    // Initialize stream and communicator
    stream_ptr_ = (cudaStream_t*)malloc(sizeof(cudaStream_t) * this->n_devices_);
    comm_ptr_ = (ncclComm_t*)malloc(sizeof(ncclComm_t) * this->n_devices_);
    ncclCommInitAll(comm_ptr_, this->n_devices_, this->device_ids_.data());

    for (int i = 0; i < n_devices_; ++i) {
      cudaSetDevice(device_ids_[i]);
      cudaStreamCreate(stream_ptr_ + i);
    }

  } catch (...) {
    this->initialized_ = false;
  }
  this->initialized_ = true;
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU reduce is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allreduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU allreduce is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reducescatter(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU reducescatter is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::bcast() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU bcast is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allgather() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU allgather is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::ireduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU ireduce is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::iallreduce(bool division) {
  // Sync all devices
  wait_by_devices_synchronization();

  // Once sync to prevent the hang where the memcpy occurs during the allreduce.
  this->sync_all_params();

  // Inpalce allreduce
  for (int i = 0; i < device_ids_.size(); ++i) {  // device-loop
    Context ctx = this->contexts_[i];
    auto device_id = device_ids_[i];
    cudaSetDevice(device_id);

    auto func_named_param = this->device_func_named_param_[i];
    auto comm = comm_ptr_[i];
    auto stream = stream_ptr_[i];
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
    }
  }
  // Divide using the same streams
  divide_by_num_divices(division);

  // Sync streams
  wait_by_streams_synchronization();
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::ireducescatter(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU ireducescatter is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::ibcast() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU ibcast is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::iallgather() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU iallgather is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allreduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allreduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU reducescatter_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::bcast_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU bcast_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU allgather_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::ireduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::iallreduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallreduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::ireducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ireducescatter_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::ibcast_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU ibcast_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::iallgather_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CPU iallgather_async is not implemented.")
}

template<typename T>
vector<string> MultiProcessDataParallelCommunicatorNccl<T>::allowed_array_classes() {
  NBLA_ERROR(error_code::not_implemented,
      "Derived class of MultiProcessDataParallelCommunicatorNccl must implement allowed_array_classes().")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::wait_by_devices_synchronization() {
  for (int i = 0; i < device_ids_.size(); ++i) {
    cudaSetDevice(device_ids_[i]);
    cudaDeviceSynchronize();
  }
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::wait_by_streams_synchronization() {
  for (int i = 0; i < device_ids_.size(); ++i) {
    cudaSetDevice(device_ids_[i]);
    cudaStreamSynchronize(stream_ptr_[i]);
  }
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::divide_by_num_divices(bool division) {
  if (division) {
    for (int i = 0; i < device_ids_.size(); ++i) {
      auto device_id = device_ids_[i];
      cudaSetDevice(device_id);

      Context ctx = this->contexts_[i];
      auto func_named_param = this->device_func_named_param_[i];
      auto stream = stream_ptr_[i];
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
void MultiProcessDataParallelCommunicatorNccl<T>::sync_all_params() {
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

template class MultiProcessDataParallelCommunicatorNccl<float>;
}
