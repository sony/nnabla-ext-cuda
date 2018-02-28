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

#include <nbla/cuda/common.hpp>
#include <nbla/cuda/communicator/multi_process_data_parallel_communicator.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>

#include "mpi.h"
#include <stdint.h>
#include <unistd.h>

namespace nbla {

using std::make_shared;

template <typename T>
__global__ void kernel_divide_inplace(const int size, const int n_devices,
                                      T *dw) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { dw[i] /= n_devices; }
}

/*
 * Referred from
 * http://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html#onedevprothrd
 */
static uint64_t get_host_hash(const char *string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++) {
    result = ((result << 5) + result) + string[c];
  }
  return result;
}

static void get_host_name(char *hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i = 0; i < maxlen; i++) {
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

template <typename T>
MultiProcessDataParallelCommunicatorNccl<
    T>::MultiProcessDataParallelCommunicatorNccl(const Context &ctx)
    : MultiProcessDataParallelCommunicator<T>(ctx) {
  mpi_initialized_ = false;
}

template <typename T>
MultiProcessDataParallelCommunicatorNccl<
    T>::~MultiProcessDataParallelCommunicatorNccl() {
  if (this->initialized_) {
    for (int i = 0; i < streams_.size(); ++i) {
      NBLA_CUDA_CHECK(cudaStreamDestroy(streams_[i]));
    }
    for (auto e : this->comms_) {
      ncclCommDestroy(e.second);
    }
  }
  if (mpi_initialized_) {
    MPI_Finalize();
  }
}

template <typename T>
bool MultiProcessDataParallelCommunicatorNccl<T>::mpi_initialized_;

template <typename T> void MultiProcessDataParallelCommunicatorNccl<T>::init() {
  Communicator::init();
  try {
    // MPI init
    if (!mpi_initialized_) {
      int argc = 0;
      char **argv = NULL;
      int requiredThreadLevelSupport = MPI_THREAD_SERIALIZED;
      int provided;
      MPI_Init_thread(&argc, &argv, requiredThreadLevelSupport, &provided);
      if (provided != requiredThreadLevelSupport) {
        NBLA_ERROR(error_code::target_specific,
                   "MPI_Init_thread failed since provided (%d) is not equal to "
                   "requiredThreadLevelSupport (%d)",
                   provided, requiredThreadLevelSupport);
      }
      mpi_initialized_ = true;
    }
    // Create comm, set size, and rank
    MPI_Comm_size(MPI_COMM_WORLD, &this->size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_);

    // Set local rank and device id
    uint64_t host_hashs[this->size_];
    char hostname[1024];
    get_host_name(hostname, 1024);
    host_hashs[this->rank_] = get_host_hash(hostname);

    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_hashs,
                  sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    int local_rank = 0;
    for (int i = 0; i < this->size_; ++i) {
      if (i == this->rank_) {
        break;
      }
      if (host_hashs[i] == host_hashs[this->rank_]) {
        local_rank++;
      }
    }
    this->device_id_ = local_rank;
    this->local_rank_ = local_rank;
    this->ctx_.device_id = std::to_string(local_rank);

    // Exchange comm_id among processes
    ncclUniqueId comm_id;
    if (this->rank_ == 0) {
      ncclGetUniqueId(&comm_id);
    }
    MPI_Bcast(&comm_id, sizeof(comm_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    // NCCL Init
    cuda_set_device(device_id_);
    ncclComm_t comm;
    ncclResult_t ret =
        ncclCommInitRank(&comm, this->size_, comm_id, this->rank_);
    if (ret != ncclSuccess) {
      NBLA_ERROR(error_code::target_specific, "ncclCommInitRank failed.");
    }

    // Create streams
    for (int i = 0; i < streams_.size(); ++i) {
      cudaStream_t stream;
      NBLA_CUDA_CHECK(cudaStreamCreate(&stream));
      streams_[i] = stream;
    }

    // Create world group
    this->comms_["world"] = comm;
    vector<int> ranks(this->size_);
    std::iota(ranks.begin(), ranks.end(), 0);
    this->groups_["world"] = ranks;

    this->initialized_ = true;
  } catch (...) {
    NBLA_ERROR(error_code::unclassified, "Communicator init failed.");
  }
}

template <typename T>
string MultiProcessDataParallelCommunicatorNccl<T>::new_group(
    pair<string, vector<int>> name_ranks_pair) {
  string group_name = name_ranks_pair.first;
  vector<int> ranks = name_ranks_pair.second;

  // Checks
  if (this->groups_.find(group_name) !=
      this->groups_.end()) { // group name already exists.
    NBLA_ERROR(error_code::value, "group_name = %s already exists",
               group_name.c_str());
  }
  int max = *std::max_element(ranks.begin(), ranks.end());
  NBLA_CHECK(max < this->size_, error_code::value,
             "Max value of the specified ranks should be less than the size () "
             "of the communicator.",
             this->size_);
  int min = *std::min_element(ranks.begin(), ranks.end());
  NBLA_CHECK(min >= 0, error_code::value,
             "Min value of the specified ranks is greater than or equal to 0.");

  // Create new group
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  MPI_Group new_group;
  MPI_Group_incl(world_group, ranks.size(), ranks.data(), &new_group);

  // Create mpi communicator
  MPI_Comm mpi_comm;
  MPI_Comm_create(MPI_COMM_WORLD, new_group,
                  &mpi_comm); // have to call in all processes

  // Add group name in all ranks
  this->groups_[group_name] = ranks;

  // Leave if self is not in ranks
  auto result = std::find(ranks.begin(), ranks.end(), this->rank_);
  if (result == ranks.end()) { // self is not found in ranks.
    return group_name;
  }

  // Create nccl unique id and bcast it
  ncclUniqueId comm_id;
  if (this->rank_ == ranks[0]) {
    ncclGetUniqueId(&comm_id);
  }
  int rank;
  MPI_Comm_rank(mpi_comm, &rank);
  MPI_Bcast(&comm_id, sizeof(comm_id), MPI_BYTE, 0, mpi_comm);
  MPI_Barrier(mpi_comm);
  MPI_Comm_free(&mpi_comm);

  // NCCL Comm Init
  cuda_set_device(device_id_);
  ncclComm_t comm;
  ncclResult_t ret = ncclCommInitRank(&comm, ranks.size(), comm_id, rank);
  if (ret != ncclSuccess) {
    NBLA_ERROR(error_code::target_specific, "ncclCommInitRank failed with %d",
               ret);
  }
  this->comms_[group_name] = comm;

  return group_name;
}

template <typename T>
shared_ptr<NdArray>
MultiProcessDataParallelCommunicatorNccl<T>::copy_inside_device(
    const vector<NdArrayPtr> &ndarray_list) {
  // preparation
  Size_t total_params = 0;
  for (auto ndarray : ndarray_list) {
    auto n_param = ndarray->size();
    total_params += n_param;
  }
  dtypes dtype = get_dtype<T>();
  NdArrayPtr large_ndarray = make_shared<NdArray>(Shape_t{total_params});
  T *buff = large_ndarray->cast(dtype, this->ctx_)->pointer<T>();
  Size_t type_size = sizeof(T);
  int k = 0;

  // copy inside device
  for (auto ndarray : ndarray_list) {
    const T *dw = ndarray->cast(dtype, this->ctx_)->const_pointer<T>();
    auto n_param = ndarray->size();
    int stream_id = k % num_streams_;
    cudaMemcpyAsync(buff, dw, type_size * n_param, cudaMemcpyDeviceToDevice,
                    streams_[stream_id]);
    buff += n_param;
    k++;
  }
  return large_ndarray;
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::copy_back_inside_device(
    const vector<NdArrayPtr> &ndarray_list, NdArrayPtr large_ndarray) {
  dtypes dtype = get_dtype<T>();
  T *buff = large_ndarray->cast(dtype, this->ctx_)->pointer<T>();
  Size_t type_size = sizeof(T);
  int k = 0;
  for (auto ndarray : ndarray_list) {
    T *dw = ndarray->cast(dtype, this->ctx_)->pointer<T>();
    auto n_param = ndarray->size();
    int stream_id = k % num_streams_;
    cudaMemcpyAsync(dw, buff, type_size * n_param, cudaMemcpyDeviceToDevice,
                    streams_[stream_id]);
    buff += n_param;
    k++;
  }
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reduce(
    const vector<NdArrayPtr> &ndarray_list, int dst, bool division,
    bool inplace, const string &group) {
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }

  // TODO: currently nnabla uses default stream for computation.
  // The following logic relies on that, so if nnabla uses another stream for
  // computation,
  // we have to issue null kernel to the default stream at the beginning of this
  // method
  // and at the end of this method for using the implicit synchronization
  // technique for
  // main thread not to wait for a result of a kernel call.

  // TODO: the usage of multi streams is round-robin fashion, it may not be
  // optimal.

  if (inplace) { // in-place
    int k = 0;
    dtypes dtype = get_dtype<T>();
    for (auto ndarray : ndarray_list) { // ndarray loop
      int stream_id = k % num_streams_;
      reduce(ndarray, streams_[stream_id], dst, division, inplace);
      k++;
    }
  } else { // out-of-place. use a large array.
    NdArrayPtr large_ndarray = copy_inside_device(ndarray_list);
    reduce(large_ndarray, nullptr, dst, division, inplace, group);
    copy_back_inside_device(ndarray_list, large_ndarray);
  }
  // no need to call null kernel since nnabla uses default stream currently.
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reduce(NdArrayPtr ndarray,
                                                         int dst, bool division,
                                                         bool inplace,
                                                         const string &group) {
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }
  reduce(ndarray, nullptr, dst, division, inplace, group);
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reduce(NdArrayPtr ndarray,
                                                         cudaStream_t stream,
                                                         int dst, bool division,
                                                         bool inplace,
                                                         const string &group) {
  auto n_param = ndarray->size();
  dtypes dtype = get_dtype<T>();
  const T *dw0 = ndarray->get(dtype, this->ctx_)->const_pointer<T>();
  T *dw1 = ndarray->cast(dtype, this->ctx_)->pointer<T>();
  ncclResult_t ret = ncclReduce(dw0, dw1, n_param,
                                ncclFloat, // TODO: address ncclFloat
                                ncclSum, dst, comms_[group], stream);
  if (ret != ncclSuccess) {
    NBLA_ERROR(error_code::target_specific, "ncclReduce fails with %d.", ret);
  }
  if (division) {
    NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, stream, n_param,
                                      this->size_, dw1);
    // TODO: strange because of implicit synchronization without inplace and
    // with division does not occur.
    // copy(streams) -> all_reduce(default stream) ->
    // -> division(default stream) -> copy_back(streams) -> xxx(default stream)
    // Even if launching null kernel, no sync. Thus, call stream synchronize.
    if (!inplace) {
      cudaStreamSynchronize(stream);
    }
  }
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allreduce(bool division,
                                                            bool inplace) {
  // TODO: Delete this function when appropriate

  // TODO: currently nnabla uses default stream for computation.
  // The following logic relies on that, so if nnabla uses another stream for
  // computation,
  // we have to issue null kernel to the default stream at the beginning of this
  // method
  // and at the end of this method for using the implicit synchronization
  // technique for
  // main thread not to wait for a result of a kernel call.

  // TODO: the usage of multi streams is round-robin fashion, it may not be
  // optimal.

  // Once sync to prevent the hang where the memcpy occurs during the allreduce.
  this->sync_all_params();

  if (inplace) { // in-place
    Context ctx = this->contexts_[0];
    auto func_named_param = this->device_func_named_param_[0];
    int k = 0;
    for (auto elm : func_named_param) { // function-loop
      VariablePtr vp = elm.second;
      auto n_param = vp->size();
      const T *dw0 = vp->get_grad_pointer<T>(ctx);
      T *dw1 = vp->cast_grad_and_get_pointer<T>(ctx);
      int stream_id = k % num_streams_;
      // AllReduce
      ncclResult_t ret =
          ncclAllReduce(dw0, dw1, n_param, ncclFloat, // TODO: address ncclFloat
                        ncclSum, comms_["world"], streams_[stream_id]);
      if (ret != ncclSuccess) {
        NBLA_ERROR(error_code::target_specific, "ncclAllReduce fails with %d.",
                   ret);
      }
      // Divide
      if (division) {
        NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace,
                                          streams_[stream_id], n_param,
                                          this->size_, dw1);
      }
      k++;
    }
  } else { // out-of-place. use a large array.
    Context ctx = this->contexts_[0];
    shared_ptr<CudaCachedArray> arr_buff = // TODO: address 16 bits also here?
        make_shared<CudaCachedArray>(this->total_params_, get_dtype<T>(), ctx);
    T *buff = arr_buff->pointer<T>();
    T *buff_start = buff;
    auto func_named_param = this->device_func_named_param_[0];
    Size_t type_size = sizeof(T);
    int k = 0;

    // 1. copy inside device
    for (auto elm : func_named_param) {
      VariablePtr vp = elm.second;
      const T *dw = vp->get_grad_pointer<T>(ctx);
      auto n_param = vp->size();
      int stream_id = k % num_streams_;
      cudaMemcpyAsync(buff, dw, type_size * n_param, cudaMemcpyDeviceToDevice,
                      streams_[stream_id]);
      buff += n_param;
      k++;
    }

    // 2. all reduce
    ncclResult_t ret =
        ncclAllReduce(buff_start, buff_start, this->total_params_,
                      ncclFloat,                    // TODO: address ncclFloat
                      ncclSum, comms_["world"], 0); // use default stream

    if (ret != ncclSuccess) {
      NBLA_ERROR(error_code::target_specific, "ncclAllReduce fails with %d.",
                 ret);
    }

    // 3. divide
    if (division) {
      // use default stream
      NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, 0,
                                        this->total_params_, this->size_,
                                        buff_start);
    }

    // 4. copy back inside device
    buff = buff_start;
    k = 0;
    for (auto elm : func_named_param) {
      VariablePtr vp = elm.second;
      T *dw = vp->cast_grad_and_get_pointer<T>(ctx);
      auto n_param = vp->size();
      int stream_id = k % num_streams_;
      cudaMemcpyAsync(dw, buff, type_size * n_param, cudaMemcpyDeviceToDevice,
                      streams_[stream_id]);
      buff += n_param;
      k++;
    }
  }
  // no need to call null kernel since nnabla uses default stream currently.
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::all_reduce(
    const vector<NdArrayPtr> &ndarray_list, bool division, bool inplace,
    const string &group) {
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }

  // TODO: currently nnabla uses default stream for computation.
  // The following logic relies on that, so if nnabla uses another stream for
  // computation,
  // we have to issue null kernel to the default stream at the beginning of this
  // method
  // and at the end of this method for using the implicit synchronization
  // technique for
  // main thread not to wait for a result of a kernel call.

  // TODO: the usage of multi streams is round-robin fashion, it may not be
  // optimal.

  if (inplace) { // in-place
    int k = 0;
    dtypes dtype = get_dtype<T>();
    for (auto ndarray : ndarray_list) { // ndarray loop
      int stream_id = k % num_streams_;
      all_reduce(ndarray, streams_[stream_id], division, inplace, group);
      k++;
    }
  } else { // out-of-place. use a large array.
    NdArrayPtr large_ndarray = copy_inside_device(ndarray_list);
    all_reduce(large_ndarray, nullptr, division, inplace, group);
    copy_back_inside_device(ndarray_list, large_ndarray);
  }
  // no need to call null kernel since nnabla uses default stream currently.
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::all_reduce(
    NdArrayPtr ndarray, bool division, bool inplace, const string &group) {
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }
  all_reduce(ndarray, nullptr, division, inplace, group);
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::all_reduce(
    NdArrayPtr ndarray, cudaStream_t stream, bool division, bool inplace,
    const string &group) {
  auto n_param = ndarray->size();
  dtypes dtype = get_dtype<T>();
  const T *dw0 = ndarray->get(dtype, this->ctx_)->const_pointer<T>();
  T *dw1 = ndarray->cast(dtype, this->ctx_)->pointer<T>();
  ncclResult_t ret = ncclAllReduce(dw0, dw1, n_param,
                                   ncclFloat, // TODO: address ncclFloat
                                   ncclSum, comms_[group], stream);
  if (ret != ncclSuccess) {
    NBLA_ERROR(error_code::target_specific, "ncclAllReduce fails with %d.",
               ret);
  }
  if (division) {
    NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, stream, n_param,
                                      this->size_, dw1);
    // TODO: strange because of implicit synchronization without inplace and
    // with division does not occur.
    // copy(streams) -> all_reduce(default stream) ->
    // -> division(default stream) -> copy_back(streams) -> xxx(default stream)
    // Even if launching null kernel, no sync. Thus, call stream synchronize.
    if (!inplace) {
      cudaStreamSynchronize(stream);
    }
  }
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reduce_scatter(
    const vector<NdArrayPtr> &ndarray_list, NdArrayPtr ndarray, bool division,
    const string &group) {
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }

  // TODO: currently nnabla uses default stream for computation.
  // The following logic relies on that, so if nnabla uses another stream for
  // computation,
  // we have to issue null kernel to the default stream at the beginning of this
  // method
  // and at the end of this method for using the implicit synchronization
  // technique for
  // main thread not to wait for a result of a kernel call.

  NdArrayPtr large_ndarray = copy_inside_device(ndarray_list);
  dtypes dtype = get_dtype<T>();
  const T *sendbuff = large_ndarray->get(dtype, this->ctx_)->const_pointer<T>();
  T *recvbuff = ndarray->cast(dtype, this->ctx_)->pointer<T>();
  Size_t recvcount = ndarray->size();
  ncclResult_t ret =
      ncclReduceScatter(sendbuff, recvbuff, recvcount,
                        ncclFloat,                  // TODO: address ncclFloat
                        ncclSum, comms_[group], 0); // use default stream
  if (ret != ncclSuccess) {
    NBLA_ERROR(error_code::target_specific, "ncclBcast fails with %d.", ret);
  }

  // divide
  if (division) {
    // use default stream
    NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, 0, recvcount,
                                      this->size_, recvbuff);
  }
  // no need to call null kernel since nnabla uses default stream currently.
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::bcast(
    const vector<NdArrayPtr> &ndarray_list, int src, bool inplace,
    const string &group) {
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }

  // TODO: currently nnabla uses default stream for computation.
  // The following logic relies on that, so if nnabla uses another stream for
  // computation,
  // we have to issue null kernel to the default stream at the beginning of this
  // method
  // and at the end of this method for using the implicit synchronization
  // technique for
  // main thread not to wait for a result of a kernel call.

  // TODO: the usage of multi streams is round-robin fashion, it may not be
  // optimal.

  if (inplace) { // in-place
    int k = 0;
    dtypes dtype = get_dtype<T>();
    for (auto ndarray : ndarray_list) { // ndarray loop
      int stream_id = k % num_streams_;
      bcast(ndarray, streams_[stream_id], src, inplace, group);
      k++;
    }
  } else { // out-of-place. use a large array.
    NdArrayPtr large_ndarray = copy_inside_device(ndarray_list);
    bcast(large_ndarray, nullptr, src, inplace, group);
    copy_back_inside_device(ndarray_list, large_ndarray);
  }
  // no need to call null kernel since nnabla uses default stream currently.
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::bcast(NdArrayPtr ndarray,
                                                        int src, bool inplace,
                                                        const string &group) {
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }
  bcast(ndarray, nullptr, src, inplace, group);
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::bcast(NdArrayPtr ndarray,
                                                        cudaStream_t stream,
                                                        int src, bool inplace,
                                                        const string &group) {
  auto n_param = ndarray->size();
  dtypes dtype = get_dtype<T>();
  T *dw0 = ndarray->cast(dtype, this->ctx_)->pointer<T>();
  ncclResult_t ret =
      ncclBcast(dw0, n_param, ncclFloat, src, comms_[group], stream);
  if (ret != ncclSuccess) {
    NBLA_ERROR(error_code::target_specific, "ncclBcast fails with %d.", ret);
  }
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::all_gather(
    NdArrayPtr ndarray, const vector<NdArrayPtr> &ndarray_list,
    const string &group) {
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }

  // TODO: currently nnabla uses default stream for computation.
  // The following logic relies on that, so if nnabla uses another stream for
  // computation,
  // we have to issue null kernel to the default stream at the beginning of this
  // method
  // and at the end of this method for using the implicit synchronization
  // technique for
  // main thread not to wait for a result of a kernel call.

  // TODO: the usage of multi streams is round-robin fashion, it may not be
  // optimal.

  NdArrayPtr large_ndarray = copy_inside_device(ndarray_list);
  dtypes dtype = get_dtype<T>();
  const T *sendbuff = ndarray->get(dtype, this->ctx_)->const_pointer<T>();
  T *recvbuff = large_ndarray->cast(dtype, this->ctx_)->pointer<T>();
  Size_t sendcount = ndarray->size();
  ncclResult_t ret = ncclAllGather(sendbuff, recvbuff, sendcount,
                                   ncclFloat,         // TODO: address ncclFloat
                                   comms_[group], 0); // use default stream
  if (ret != ncclSuccess) {
    NBLA_ERROR(error_code::target_specific, "ncclAllGather fails with %d.",
               ret);
  }
  copy_back_inside_device(ndarray_list, large_ndarray);
  // no need to call null kernel since nnabla uses default stream currently.
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reduce_async(bool division) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU reduce_async is not implemented.")
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allreduce_async(
    bool division, bool inplace) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU allreduce_async is not implemented.")
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::reducescatter_async(
    bool division) {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU reducescatter_async is not implemented.")
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::bcast_async() {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU bcast_async is not implemented.")
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::allgather_async() {
  NBLA_ERROR(error_code::not_implemented,
             "CUDA GPU allgather_async is not implemented.")
}

template <typename T>
vector<string>
MultiProcessDataParallelCommunicatorNccl<T>::allowed_array_classes() {
  NBLA_ERROR(error_code::not_implemented,
             "Derived class of MultiProcessDataParallelCommunicatorNccl must "
             "implement allowed_array_classes().")
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<
    T>::wait_by_device_synchronization() {
  cuda_device_synchronize(device_id_);
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<
    T>::wait_by_streams_synchronization() {
  for (int i = 0; i < streams_.size(); ++i) {
    NBLA_CUDA_CHECK(cudaStreamSynchronize(streams_[i]));
  }
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::sync_all_params() {
  auto func_named_param = this->device_func_named_param_[0];
  Context ctx = this->contexts_[0];
  auto size = func_named_param.size();

  for (auto elm : func_named_param) { // function-loop
    VariablePtr vp = elm.second;

    // If the arrays are different, output the warning.
    this->check_array_class(ctx, vp);

    // Sync
    vp->get_grad_pointer<T>(ctx);
  }
}

template class MultiProcessDataParallelCommunicatorNccl<float>;
}
