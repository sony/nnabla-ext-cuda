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

#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/common.hpp>
#include <nbla/cuda/communicator/data_parallel_communicator.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>

namespace nbla {

using std::make_shared;

template <typename T>
__global__ void kernel_divide_inplace(const int size, const int n_devices,
                                      T *dw) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { dw[i] /= n_devices; }
}

template <typename T>
DataParallelCommunicatorNccl<T>::DataParallelCommunicatorNccl(
    const Context &ctx)
    : DataParallelCommunicator<T>(ctx) {}

template <typename T>
DataParallelCommunicatorNccl<T>::~DataParallelCommunicatorNccl() {
  if (this->initialized_) {
    for (int i = 0; i < device_ids_.size(); ++i) {
      ncclCommDestroy(comms_[i]);
      NBLA_CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
  }
}

template <typename T> void DataParallelCommunicatorNccl<T>::init() {
  Communicator::init();
  try {
    // Set gpu information
    for (auto ctx : this->contexts_) {
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
    ncclResult_t res = ncclCommInitAll(comms_.data(), this->n_devices_,
                                       this->device_ids_.data());
    if (res != 0) {
      NBLA_ERROR(error_code::target_specific, "ncclCommInitAll fails with %d");
    }
  } catch (...) {
    this->initialized_ = false;
  }
  this->initialized_ = true;
}

template <typename T>
void DataParallelCommunicatorNccl<T>::reduce(
    const vector<NdArrayPtr> &ndarray_list, int dst, bool division,
    bool inplace, const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CUDA GPU reduce is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::reduce(NdArrayPtr ndarray, int dst,
                                             bool division, bool inplace,
                                             const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CUDA GPU reduce is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::allreduce(bool division, bool inplace) {
  // TODO: currently nnabla uses default stream for computation.
  // The following logic relies on that, so if nnabla uses another stream for
  // computation,
  // we have to issue null kernel to the default stream at the beginning of this
  // method
  // and at the end of this method for using the implicit synchronization
  // technique for
  // main thread not to wait for a result of a kernel call.

  if (inplace == true) {
    NBLA_ERROR(error_code::not_implemented,
               "CUDA GPU allreduce with out-of-place is only implemented.")
  }

  // Once sync to prevent the hang where the memcpy occurs during the allreduce.
  this->sync_all_params();

  // 1. copy inside device
  for (int i = 0; i < device_ids_.size(); ++i) { // device-loop
    Context ctx = this->contexts_[i];
    auto device_id = device_ids_[i];
    cuda_set_device(device_id);

    auto func_named_param = this->device_func_named_param_[i];
    auto comm = comms_[i];
    auto stream = streams_[i];

    shared_ptr<CudaCachedArray> arr_buff = // TODO: address 16 bits also here?
        make_shared<CudaCachedArray>(this->total_params_, get_dtype<Tc>(), ctx);

    Tc *buff = arr_buff->pointer<Tc>();
    Size_t type_size = sizeof(Tc);

    for (auto elm : func_named_param) {
      VariablePtr vp = elm.second;
      const Tc *dw = vp->get_grad_pointer<Tc>(ctx);
      auto n_param = vp->size();
      cudaMemcpyAsync(buff, dw, type_size * n_param, cudaMemcpyDeviceToDevice,
                      stream);
      buff += n_param;
    }
  }

// 2. allreduce
#ifdef NCCL_MAJOR
  ncclGroupStart();
#endif
  for (int i = 0; i < device_ids_.size(); ++i) { // device-loop
    Context ctx = this->contexts_[i];
    auto device_id = device_ids_[i];
    //    cuda_set_device(device_id);

    auto comm = comms_[i];
    auto stream = streams_[i];

    shared_ptr<CudaCachedArray> arr_buff = // TODO: address 16 bits also here?
        make_shared<CudaCachedArray>(this->total_params_, get_dtype<Tc>(), ctx);

    Tc *buff = arr_buff->pointer<Tc>();
    ncclResult_t ret =
        ncclAllReduce(buff, buff, this->total_params_, get_nccl_dtype<Tc>(),
                      ncclSum, comm, 0); // use default stream

    if (ret != ncclSuccess) {
      NBLA_ERROR(error_code::target_specific, "ncclAllReduce fails with %d.",
                 ret);
    }
  }
#ifdef NCCL_MAJOR
  ncclGroupEnd();
//  wait_by_streams_synchronization();
#endif

  // 3. divide
  if (division) {
    for (int i = 0; i < device_ids_.size(); ++i) { // device-loop
      Context ctx = this->contexts_[i];
      auto device_id = device_ids_[i];
      cuda_set_device(device_id);

      auto comm = comms_[i];
      auto stream = streams_[i];

      shared_ptr<CudaCachedArray> arr_buff = // TODO: address 16 bits also here?
          make_shared<CudaCachedArray>(this->total_params_, get_dtype<Tc>(),
                                       ctx);

      Tc *buff = arr_buff->pointer<Tc>();
      NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, stream,
                                        this->total_params_, n_devices_, buff);
    }
  }

  // 4. copy back inside device
  for (int i = 0; i < device_ids_.size(); ++i) { // device-loop
    Context ctx = this->contexts_[i];
    auto device_id = device_ids_[i];
    cuda_set_device(device_id);

    auto func_named_param = this->device_func_named_param_[i];
    auto comm = comms_[i];
    auto stream = streams_[i];

    shared_ptr<CudaCachedArray> arr_buff = // TODO: address 16 bits also here?
        make_shared<CudaCachedArray>(this->total_params_, get_dtype<Tc>(), ctx);

    Tc *buff = arr_buff->pointer<Tc>();
    Size_t type_size = sizeof(Tc);

    for (auto elm : func_named_param) {
      VariablePtr vp = elm.second;
      Tc *dw = vp->cast_grad_and_get_pointer<Tc>(ctx);
      auto n_param = vp->size();
      cudaMemcpyAsync(dw, buff, type_size * n_param, cudaMemcpyDeviceToDevice,
                      stream);
      buff += n_param;
    }
  }
}

template <typename T>
void DataParallelCommunicatorNccl<T>::all_reduce(
    const vector<NdArrayPtr> &ndarray_list, bool division, bool inplace,
    const string &group) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU all_reduce is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::all_reduce(NdArrayPtr ndarray,
                                                 bool division, bool inplace,
                                                 const string &group) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU all_reduce is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::reduce_scatter(
    const vector<NdArrayPtr> &ndarray_list, NdArrayPtr ndarray, bool division,
    const string &group) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU reduce_scatter is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::bcast(
    const vector<NdArrayPtr> &ndarray_list, int src, bool inplace,
    const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CUDA GPU bcast is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::bcast(NdArrayPtr ndarray, int src,
                                            bool inplace, const string &group) {
  NBLA_ERROR(error_code::not_implemented, "CUDA GPU bcast is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::all_gather(
    NdArrayPtr ndarray, const vector<NdArrayPtr> &ndarray_list,
    const string &group) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU all_gather is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU reduce_async is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::allreduce_async(bool division,
                                                      bool inplace) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU allreduce_async is not implemented.")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::reducescatter_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU reducescatter_async is not implemented.")
}

template <typename T> void DataParallelCommunicatorNccl<T>::bcast_async() {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU bcast_async is not implemented.")
}

template <typename T> void DataParallelCommunicatorNccl<T>::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU allgather_async is not implemented.")
}

template <typename T>
vector<string> DataParallelCommunicatorNccl<T>::allowed_array_classes() {
  NBLA_ERROR(error_code::not_implemented, "Derived class of "
                                          "DataParallelCommunicatorNccl must "
                                          "implement allowed_array_classes().")
}

template <typename T>
void DataParallelCommunicatorNccl<T>::wait_by_devices_synchronization() {
  for (int i = 0; i < device_ids_.size(); ++i) {
    cuda_device_synchronize(std::to_string(device_ids_[i]));
  }
}

template <typename T>
void DataParallelCommunicatorNccl<T>::wait_by_streams_synchronization() {
  for (int i = 0; i < device_ids_.size(); ++i) {
    cuda_set_device(device_ids_[i]);
    NBLA_CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
  }
}

template <typename T>
void DataParallelCommunicatorNccl<T>::divide_by_num_devices(bool division) {
  if (division) {
    for (int i = 0; i < device_ids_.size(); ++i) {
      auto device_id = device_ids_[i];
      cuda_set_device(device_id);

      Context ctx = this->contexts_[i];
      auto func_named_param = this->device_func_named_param_[i];
      auto stream = streams_[i];
      for (auto elm : func_named_param) {
        VariablePtr vp = elm.second;
        Tc *dw = vp->cast_grad_and_get_pointer<Tc>(ctx);
        auto n_param = vp->size();
        NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, stream,
                                          n_param, n_devices_, dw);
      }
    }
  }
}

template <typename T> void DataParallelCommunicatorNccl<T>::sync_all_params() {
  for (int i = 0; i < device_ids_.size(); ++i) { // device-loop
    Context ctx = this->contexts_[i];
    auto device_id = device_ids_[i];

    auto func_named_param = this->device_func_named_param_[i];
    auto size = func_named_param.size();

    for (auto elm : func_named_param) { // function-loop
      VariablePtr vp = elm.second;

      // If the arrays are different, output the warning.
      this->check_array_class(ctx, vp);

      // Sync
      vp->get_grad_pointer<Tc>(ctx);
    }
  }
}

template class DataParallelCommunicatorNccl<float>;
template class DataParallelCommunicatorNccl<Half>;
}
