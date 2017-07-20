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

#include "mpi.h"

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
MultiProcessDataParallelCommunicatorNccl<T>::MultiProcessDataParallelCommunicatorNccl(const Context &ctx) : MultiProcessDataParallelCommunicator<T>(ctx) {
  mpi_initialized_ = false;
}

template<typename T>
MultiProcessDataParallelCommunicatorNccl<T>::~MultiProcessDataParallelCommunicatorNccl() {
  if (this->initialized_) {
    ncclCommDestroy(comm_);
    cudaStreamDestroy(stream_);
  }
  if (mpi_initialized_) {
    MPI_Finalize();
  }
}

template<typename T>
bool MultiProcessDataParallelCommunicatorNccl<T>::mpi_initialized_;

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::init() {
  Communicator::init();
  try {
    // MPI init
    if(!mpi_initialized_) {
      int argc = 0;
      char **argv = NULL;
      int requiredThreadLevelSupport = MPI_THREAD_SERIALIZED;
      int provided;
      MPI_Init_thread(&argc, &argv, requiredThreadLevelSupport, &provided);
      if (provided != requiredThreadLevelSupport)
          NBLA_ERROR(error_code::target_specific, "MPI_Init_thread failed.");
      mpi_initialized_ = true;
    }
    // Create comm, set size, and rank
    MPI_Comm mpi_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
    MPI_Comm_size(mpi_comm, &this->size_);
    MPI_Comm_rank(mpi_comm, &this->rank_);
    device_id_ = this->rank_;

    // We have to set our device before NCCL init
    cudaSetDevice(device_id_);
    MPI_Barrier(mpi_comm);

    // Exchange comm_id_ among processes
    ncclGetUniqueId(&comm_id_);
    MPI_Bcast(&comm_id_, NCCL_UNIQUE_ID_BYTES, MPI_CHAR, 0, mpi_comm);
    MPI_Comm_free(&mpi_comm);

    // Nccl Init
    ncclResult_t ret = ncclCommInitRank(&comm_, this->size_, comm_id_, this->rank_);
    if (ret != ncclSuccess) {
      NBLA_ERROR(error_code::target_specific, "ncclCommInitRank failed.");
    }

    // Create stream
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

    this->initialized_ = true;
  } catch (...) {
    NBLA_ERROR(error_code::unclassified, "Communicator init failed.");
  }
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reduce(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU ireduce is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allreduce(bool division) {
  // Sync all devices
  wait_by_device_synchronization();

  // Once sync to prevent the hang where the memcpy occurs during the allreduce.
  this->sync_all_params();

  // Inpalce allreduce
  Context ctx = this->contexts_[0];

  auto func_named_param = this->device_func_named_param_[0];
  auto size = func_named_param.size();

  for (auto elm : func_named_param) {  // function-loop
    VariablePtr vp = elm.second;
    auto n_param = vp->size();

    const T *dw0 = vp->get_grad_pointer<T>(ctx);
    T *dw1 = vp->cast_grad_and_get_pointer<T>(ctx);
    ncclResult_t res = ncclAllReduce(
        dw0, dw1,
        n_param, ncclFloat, ncclSum, //TODO: address ncclFloat
        comm_,
        stream_);
  }

  // Divide using the same streams
  divide_by_num_divices(division);

  // Sync streams
  wait_by_stream_synchronization();
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
void MultiProcessDataParallelCommunicatorNccl<T>::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU reduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allreduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU allreduce_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU reducescatter_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::bcast_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU bcast_async is not implemented.")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
      "CUDA GPU allgather_async is not implemented.")
}

template<typename T>
vector<string> MultiProcessDataParallelCommunicatorNccl<T>::allowed_array_classes() {
  NBLA_ERROR(error_code::not_implemented,
      "Derived class of MultiProcessDataParallelCommunicatorNccl must implement allowed_array_classes().")
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::wait_by_device_synchronization() {
  cudaDeviceSynchronize();
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::wait_by_stream_synchronization() {
  cudaStreamSynchronize(stream_);
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::divide_by_num_divices(bool division) {
  if (division) {
    Context ctx = this->contexts_[0];
    auto func_named_param = this->device_func_named_param_[0];
    for (auto elm : func_named_param) {
      VariablePtr vp = elm.second;
      T *dw = vp->cast_grad_and_get_pointer<T>(ctx);
      auto n_param = vp->size();
      NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(
          kernel_divide_inplace, stream_, n_param, this->size_, dw);
    }
  }
}

template<typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::sync_all_params() {
  Context ctx = this->contexts_[0];
  auto func_named_param = this->device_func_named_param_[0];
  auto size = func_named_param.size();

  for (auto elm : func_named_param) {          // function-loop
    VariablePtr vp = elm.second;

    // If the arrays are different, output the warning.
    this->check_array_class(ctx, vp);

    // Sync
    vp->get_grad_pointer<T>(ctx);
  }
}

template class MultiProcessDataParallelCommunicatorNccl<float>;
}
