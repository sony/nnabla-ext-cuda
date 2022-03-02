// Copyright 2017,2018,2019,2020,2021 Sony Corporation.
// Copyright 2021 Sony Group Corporation.
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
#include <nbla/cuda/cuda.hpp>
#include <nbla/singleton_manager.hpp>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <nbla/cuda/communicator/dl_mpi.h>
#include <numeric>
#include <stdint.h>
#include <unistd.h>

#include <typeinfo>

namespace nbla {

using std::vector;
using std::make_shared;
using std::unordered_set;

/** Get MPI error string from error code.
 */
std::string get_mpi_error_string(int error) {
  int len;
  char estring[MPI_MAX_ERROR_STRING];
  MPI_Error_string(error, estring, &len);
  return std::string(estring);
}

/** MPI error handler which throws an exception
*/
#define NBLA_MPI_CHECK(condition)                                              \
  {                                                                            \
    int error = condition;                                                     \
    if (error != MPI_SUCCESS) {                                                \
      auto estring = get_mpi_error_string(error);                              \
      NBLA_ERROR(error_code::runtime, "`" #condition "` failed by `%s`.",      \
                 estring.c_str());                                             \
    }                                                                          \
  }

/**
   MPI singleton class that manages lifetime of MPI.

   The MPI singleton object
   can be obtained as a shared pointer by a static function get(). Using a
   shared pointer ensures the singleton is alive as long as it is holded by any
   object.

   @todo This is only used by NCCL communicator so far. However, this should be
   exposed to another communicator which relies on MPI.
 */
class Mpi {
public:
  /**
    Returns the singleton object as a shared pointer.
  */
  static shared_ptr<Mpi> get() {
    static std::shared_ptr<Mpi> mpi;
    if (!mpi) {
      mpi.reset(new Mpi);
    }
    return mpi;
  }
  /**
    Returns MPI_Group of MPI_COMM_WORLD.

    It is maintained in this singleton to ensure the MPI_Group_free is called
    when the program ends.
  */
  static MPI_Group world_group() { return Mpi::get()->world_group_; }
  ~Mpi() {
    if (finalized()) {
      /*
        MPI might be finalized by other libraries.
       */
      return;
    }
    NBLA_MPI_CHECK(MPI_Group_free(&world_group_));
    NBLA_MPI_CHECK(MPI_Finalize());
  }

  /**
     Returns whether MPI is initialzed or not.
   */
  static bool initialized() {
    int flag = 1;
    NBLA_MPI_CHECK(MPI_Initialized(&flag));
    return bool(flag);
  }

  /**
     Returns whether MPI is finalized or not.
   */
  static bool finalized() {
    int flag = 1;
    NBLA_MPI_CHECK(MPI_Finalized(&flag));
    return bool(flag);
  }

  /**
     Return whether MPI is active (initialized and not finalized).
   */
  static bool active() { return initialized() && !finalized(); }

  // Prohibit copy and move
  Mpi(const Mpi &rhs) = delete;
  Mpi(Mpi &&rhs) = delete;

private:
  /*
    Constructor is hidden to make it a singleton.
   */
  Mpi() {
    if (!initialized()) {
      /*
        Initialize if MPI haven't been initialized by other libraries.
       */
      int argc = 0;
      char **argv = nullptr;
      int requiredThreadLevelSupport = MPI_THREAD_SERIALIZED;
      int provided;
      NBLA_MPI_CHECK(
          MPI_Init_thread(&argc, &argv, requiredThreadLevelSupport, &provided));
      if (provided != requiredThreadLevelSupport) {
        NBLA_ERROR(error_code::target_specific,
                   "MPI_Init_thread failed since provided (%d) is not equal to "
                   "requiredThreadLevelSupport (%d)",
                   provided, requiredThreadLevelSupport);
      }
    }
    NBLA_MPI_CHECK(MPI_Comm_group(MPI_COMM_WORLD, &world_group_));
  }
  MPI_Group world_group_;
};

/**
  A wrapper class of MPI_Comm.

  It holds instances of MPI_Comm and MPI_Group, and those will be properly freed
  at the end of lifetime of this class object.

  @todo Move it to a public header after some refactoring.
 */
class MpiCommWrapper {
private:
  // Holds MPI singleton to ensure the life time of it
  shared_ptr<Mpi> mpi_;
  /*
    Whether it owns MPI_Comm and MPI_Group. The comm_ and group_ will be freed
    if they are owned by this class object.
  */
  bool own_;
  MPI_Comm comm_;
  MPI_Group group_;

public:
  /**
     Create an object for MPI_COMM_WORLD.
   */
  MpiCommWrapper()
      : mpi_(Mpi::get()), own_(false), comm_(MPI_COMM_WORLD),
        group_(Mpi::world_group()) {}

  /**
     Create an object for a new group given by rank integers under
     MPI_COMM_WORLD.
   */
  MpiCommWrapper(std::vector<int> ranks) : mpi_(Mpi::get()), own_(true) {
    NBLA_MPI_CHECK(MPI_Group_incl(Mpi::world_group(), ranks.size(),
                                  ranks.data(), &group_));
    NBLA_MPI_CHECK(
        MPI_Comm_create_group(MPI_COMM_WORLD, group_, 0, &this->comm_));
  }
  /*
    Deletes MPI_Group and MPI_Comm objects when it owns.
   */
  ~MpiCommWrapper() {
    if (!this->own_) {
      return;
    }
    MPI_Group_free(&group_);
    if (this->comm_ == MPI_COMM_NULL) {
      return;
    }
    MPI_Comm_free(&this->comm_);
  }
  /**
     Returns MPI_Comm holded by this object.

     It must not be freed outside.
   */
  MPI_Comm comm() { return comm_; }
  /**
     Returns MPI_Group holded by this object.

     It must not be freed outside.
   */
  MPI_Group group() { return group_; }

  // Prohibit copy and move
  MpiCommWrapper(const MpiCommWrapper &rhs) = delete;
  MpiCommWrapper(MpiCommWrapper &&rhs) = delete;
};

template <typename T>
__global__ void kernel_divide_inplace(const int size, const int n_devices,
                                      T *dw) {
  NBLA_CUDA_KERNEL_LOOP(i, size) { dw[i] /= n_devices; }
}

// TODO: move to cuda/utils?
__global__ void kernel_null() {}

inline void launch_kernel_null() { kernel_null<<<1, 1>>>(); }

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
bool MultiProcessDataParallelCommunicatorNccl<T>::mpi_check_any(
    bool condition, const string &group) {
  bool result;
  NBLA_MPI_CHECK(MPI_Allreduce(&condition, &result, 1, MPI_C_BOOL, MPI_LOR,
                               this->mpi_comms_[group]->comm()));
  return result;
}

template <typename T>
bool MultiProcessDataParallelCommunicatorNccl<T>::mpi_check_all(
    bool condition, const string &group) {
  bool result;
  NBLA_MPI_CHECK(MPI_Allreduce(&condition, &result, 1, MPI_C_BOOL, MPI_LAND,
                               this->mpi_comms_[group]->comm()));
  return result;
}

template <typename T>
vector<NdArrayPtr>
MultiProcessDataParallelCommunicatorNccl<T>::get_modified_arrays(
    const vector<NdArrayPtr> &arrays, const string &group) {
  // Get arrays which are modified after zeroing.
  vector<NdArrayPtr> modified_array_list;

  for (auto &a : arrays) {
    if (mpi_check_any(!a->array()->zeroing(), group)) {
      modified_array_list.push_back(a);
    }
  }
  return modified_array_list;
}

template <typename T>
MultiProcessDataParallelCommunicatorNccl<
    T>::MultiProcessDataParallelCommunicatorNccl(const Context &ctx)
    : MultiProcessDataParallelCommunicator<T>(ctx), watch_dog_() {}

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
    for (auto &stream : this->nonblocking_streams_) {
      NBLA_CUDA_CHECK(cudaStreamDestroy(stream));
    }
  }
}

template <typename T> void MultiProcessDataParallelCommunicatorNccl<T>::init() {
  Watchdog::WatchdogLock lck(
      watch_dog_); // check if this function is finished within 10s.
  Communicator::init();
  Mpi::get(); // Make sure MPI singleton is initialized.
  this->mpi_comms_["world"] = make_shared<MpiCommWrapper>();

  // Create comm, set size, and rank
  NBLA_MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &this->size_));
  NBLA_MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &this->rank_));

  // Set local rank and device id
  uint64_t host_hashs[this->size_];
  char hostname[1024];
  get_host_name(hostname, 1024);
  host_hashs[this->rank_] = get_host_hash(hostname);

  NBLA_MPI_CHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host_hashs,
                               sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
  NBLA_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

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
  NBLA_MPI_CHECK(
      MPI_Bcast(&comm_id, sizeof(comm_id), MPI_BYTE, 0, MPI_COMM_WORLD));
  NBLA_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));

  // NCCL Init
  cuda_set_device(device_id_);
  ncclComm_t comm;
  ncclResult_t ret = ncclCommInitRank(&comm, this->size_, comm_id, this->rank_);
  if (ret != ncclSuccess) {
    NBLA_ERROR(error_code::target_specific, "ncclCommInitRank failed.");
  }

  // Create streams
  for (int i = 0; i < streams_.size(); ++i) {
    cudaStream_t stream;
    NBLA_CUDA_CHECK(cudaStreamCreate(&stream));
    streams_[i] = stream;
  }
  for (auto &stream : this->nonblocking_streams_) {
    NBLA_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  // Create world group
  this->comms_["world"] = comm;
  vector<int> ranks(this->size_);
  std::iota(ranks.begin(), ranks.end(), 0);
  this->groups_["world"] = ranks;

  this->initialized_ = true;
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::barrier() {
  NBLA_MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::abort() {
  NBLA_MPI_CHECK(MPI_Abort(MPI_COMM_WORLD, -1));
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

  // Create mpi communicator
  auto group_mpi_comm = make_shared<MpiCommWrapper>(ranks);

  // Add group name in all ranks
  this->groups_[group_name] = ranks;

  // Add group MPI comm
  this->mpi_comms_[group_name] = group_mpi_comm;

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
  NBLA_MPI_CHECK(MPI_Comm_rank(group_mpi_comm->comm(), &rank));
  NBLA_MPI_CHECK(MPI_Bcast(&comm_id, sizeof(comm_id), MPI_BYTE, 0,
                           group_mpi_comm->comm()));
  NBLA_MPI_CHECK(MPI_Barrier(group_mpi_comm->comm()));

  // NCCL Comm Init
  cuda_set_device(device_id_);
  ncclComm_t comm;
  NBLA_NCCL_CHECK(ncclCommInitRank(&comm, ranks.size(), comm_id, rank));
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
  dtypes dtype = get_dtype<Tc>();
  NdArrayPtr large_ndarray = make_shared<NdArray>(Shape_t{total_params});
  Tc *buff = large_ndarray->cast(dtype, this->ctx_)->pointer<Tc>();
  Size_t type_size = sizeof(Tc);
  int k = 0;

  // copy inside device
  for (auto ndarray : ndarray_list) {
    const Tc *dw = ndarray->cast(dtype, this->ctx_)->const_pointer<Tc>();
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
  dtypes dtype = get_dtype<Tc>();
  Tc *buff = large_ndarray->cast(dtype, this->ctx_)->pointer<Tc>();
  Size_t type_size = sizeof(Tc);
  int k = 0;
  for (auto ndarray : ndarray_list) {
    Tc *dw = ndarray->cast(dtype, this->ctx_)->pointer<Tc>();
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

  launch_kernel_null();
  if (inplace) { // in-place
    int k = 0;
    dtypes dtype = get_dtype<Tc>();
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
  launch_kernel_null();
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
  dtypes dtype = get_dtype<Tc>();
  const Tc *dw0 = ndarray->get(dtype, this->ctx_)->const_pointer<Tc>();
  Tc *dw1 = ndarray->cast(dtype, this->ctx_)->pointer<Tc>();
  NBLA_NCCL_CHECK(ncclReduce(dw0, dw1, n_param, get_nccl_dtype<Tc>(), ncclSum,
                             dst, comms_[group], stream));
  if (division && this->rank_ == dst) {
    NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, stream, n_param,
                                      this->groups_[group].size(), dw1);
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
  launch_kernel_null();
  if (inplace) { // in-place
    Context ctx = this->contexts_[0];
    auto func_named_param = this->device_func_named_param_[0];
    int k = 0;
    for (auto elm : func_named_param) { // function-loop
      VariablePtr vp = elm.second;
      auto n_param = vp->size();
      const Tc *dw0 = vp->get_grad_pointer<Tc>(ctx);
      Tc *dw1 = vp->cast_grad_and_get_pointer<Tc>(ctx);
      int stream_id = k % num_streams_;
      // AllReduce

      NBLA_NCCL_CHECK(ncclAllReduce(dw0, dw1, n_param, get_nccl_dtype<Tc>(),
                                    ncclSum, comms_["world"],
                                    streams_[stream_id]));
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
    // TODO: address 16 bits also here?
    NdArray arr_buff(Shape_t{this->total_params_});
    Tc *buff = arr_buff.cast(get_dtype<Tc>(), ctx, true)->pointer<Tc>();
    Tc *buff_start = buff;
    auto func_named_param = this->device_func_named_param_[0];
    Size_t type_size = sizeof(Tc);
    int k = 0;

    // 1. copy inside device
    for (auto elm : func_named_param) {
      VariablePtr vp = elm.second;
      const Tc *dw = vp->get_grad_pointer<Tc>(ctx);
      auto n_param = vp->size();
      int stream_id = k % num_streams_;
      cudaMemcpyAsync(buff, dw, type_size * n_param, cudaMemcpyDeviceToDevice,
                      streams_[stream_id]);
      buff += n_param;
      k++;
    }

    // 2. all reduce
    NBLA_NCCL_CHECK(ncclAllReduce(buff_start, buff_start, this->total_params_,
                                  get_nccl_dtype<Tc>(), ncclSum,
                                  comms_["world"],
                                  0)); // use default stream

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
      Tc *dw = vp->cast_grad_and_get_pointer<Tc>(ctx);
      auto n_param = vp->size();
      int stream_id = k % num_streams_;
      cudaMemcpyAsync(dw, buff, type_size * n_param, cudaMemcpyDeviceToDevice,
                      streams_[stream_id]);
      buff += n_param;
      k++;
    }
  }
  launch_kernel_null();
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::all_reduce(
    const vector<NdArrayPtr> &ndarray_list, bool division, bool inplace,
    const string &group) {
  // If it is rank0, we check all other nodes with strict constraint, for
  // example,
  // must be finished within 50s. Otherwise, we loose this constraint to allow
  // rank0 to do a relative long time operation.
  int timeout = this->rank_ == 0 ? all_reduce_timeout_
                                 : all_reduce_timeout_ * TIMES_FOR_OTHER_NODE;
  Watchdog::WatchdogLock lck(watch_dog_, timeout);
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

  launch_kernel_null();
  if (inplace) { // in-place
    int k = 0;
    dtypes dtype = get_dtype<Tc>();
    for (auto ndarray : ndarray_list) { // ndarray loop
      if (mpi_check_all(ndarray->array()->zeroing(), group)) {
        continue;
      }
      int stream_id = k % num_streams_;
      all_reduce(ndarray, streams_[stream_id], division, inplace, group);
      k++;
    }
  } else { // out-of-place. use a large array.
    auto modified_ndarray_list = get_modified_arrays(ndarray_list, group);
    if (!modified_ndarray_list.empty()) {
      NdArrayPtr large_ndarray = copy_inside_device(modified_ndarray_list);
      all_reduce(large_ndarray, nullptr, division, inplace, group);
      copy_back_inside_device(modified_ndarray_list, large_ndarray);
    }
  }
  launch_kernel_null();
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::all_reduce(
    NdArrayPtr ndarray, bool division, bool inplace, const string &group) {
  Watchdog::WatchdogLock lck(watch_dog_, all_reduce_timeout_);
  if (!this->find_self(group)) {
    NBLA_ERROR(error_code::value, "self (rank=%d) is not included in %s.",
               this->rank_, group.c_str());
  }
  if (mpi_check_all(ndarray->array()->zeroing(), group)) {
    // Skip since this array is not updated.
    return;
  }
  all_reduce(ndarray, nullptr, division, inplace, group);
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::all_reduce(
    NdArrayPtr ndarray, cudaStream_t stream, bool division, bool inplace,
    const string &group) {
  auto n_param = ndarray->size();
  dtypes dtype = get_dtype<Tc>();
  Tc *dw = ndarray->cast(dtype, this->ctx_)->pointer<Tc>();
  this->all_reduce(dw, n_param, stream, division, inplace, group);
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::all_reduce(
    Tc *gpu_buffer, size_t n_param, cudaStream_t stream, bool division,
    bool inplace, const string &group) {
  NBLA_NCCL_CHECK(ncclAllReduce(gpu_buffer, gpu_buffer, n_param,
                                get_nccl_dtype<Tc>(), ncclSum,
                                this->comms_[group], stream));
  if (division) {
    NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, stream, n_param,
                                      this->groups_[group].size(), gpu_buffer);
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

  launch_kernel_null();
  NdArrayPtr large_ndarray = copy_inside_device(ndarray_list);
  dtypes dtype = get_dtype<Tc>();
  const Tc *sendbuff =
      large_ndarray->get(dtype, this->ctx_)->const_pointer<Tc>();
  Tc *recvbuff = ndarray->cast(dtype, this->ctx_)->pointer<Tc>();
  Size_t recvcount = ndarray->size();
  NBLA_NCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recvcount,
                                    get_nccl_dtype<Tc>(), ncclSum,
                                    comms_[group],
                                    0)); // use default stream

  // divide
  if (division) {
    // use default stream
    NBLA_CUDA_LAUNCH_KERNEL_IN_STREAM(kernel_divide_inplace, 0, recvcount,
                                      this->groups_[group].size(), recvbuff);
  }
  launch_kernel_null();
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

  launch_kernel_null();
  if (inplace) { // in-place
    int k = 0;
    dtypes dtype = get_dtype<Tc>();
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
  launch_kernel_null();
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
  dtypes dtype = get_dtype<Tc>();
  Tc *dw0 = ndarray->cast(dtype, this->ctx_)->pointer<Tc>();
  NBLA_NCCL_CHECK(ncclBcast(dw0, n_param, get_nccl_dtype<Tc>(), src,
                            comms_[group], stream));
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
  launch_kernel_null();
  NdArrayPtr large_ndarray = copy_inside_device(ndarray_list);
  dtypes dtype = get_dtype<Tc>();
  const Tc *sendbuff = ndarray->get(dtype, this->ctx_)->const_pointer<Tc>();
  Tc *recvbuff = large_ndarray->cast(dtype, this->ctx_)->pointer<Tc>();
  Size_t sendcount = ndarray->size();
  NBLA_NCCL_CHECK(ncclAllGather(sendbuff, recvbuff, sendcount,
                                get_nccl_dtype<Tc>(), comms_[group],
                                0)); // use default stream
  copy_back_inside_device(ndarray_list, large_ndarray);
  launch_kernel_null();
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
CommunicatorBackwardCallbackPtr
MultiProcessDataParallelCommunicatorNccl<T>::all_reduce_callback(
    const vector<NdArrayPtr> &ndarray_list, size_t pack_size, bool division,
    const string &group) {
  dtypes dtype = get_dtype<Tc>();

  /* Allocate GPU memory(buffers) for packing. */
  /* Limit the number of buffer in order to limit the memory usage
   *   If pack_size < max_gpu_memory_size: the memory usage is less than or
   *                                       equal to max_gpu_memory_size.
   *   Otherwise                         : the memory usage is equal pack_size.
   */
  auto n = (max_gpu_memory_size < pack_size)
               ? 1
               : max_gpu_memory_size / sizeof(Tc) / pack_size;
  auto gpu_memory =
      make_shared<NdArray>(Shape_t{static_cast<int>(n * pack_size)});

  /* Create a hash-set that contains all pointers to all-reduce */
  unordered_set<NdArrayPtr> device_ptrs(ndarray_list.begin(),
                                        ndarray_list.end());
  return make_shared<AllReduceCallback>(*this, group, pack_size, division,
                                        gpu_memory, device_ptrs);
}

template <typename T>
CommunicatorBackwardCallbackPtr
MultiProcessDataParallelCommunicatorNccl<T>::all_reduce_callback(
    NdArrayPtr ndarray, size_t pack_size, bool division, const string &group) {
  /* Not implemented here, only for removing warning... */
  return nullptr;
}

template <typename T>
MultiProcessDataParallelCommunicatorNccl<T>::AllReduceCallback::
    AllReduceCallback(MultiProcessDataParallelCommunicatorNccl<T> &parent,
                      const string &group, size_t n_params_threshold,
                      bool division, const NdArrayPtr &gpu_memory,
                      const unordered_set<NdArrayPtr> &device_ptrs)
    : parent_(parent), group_(group), n_params_threshold_(n_params_threshold),
      division_(division), gpu_memory_(gpu_memory), device_ptrs_(device_ptrs),
      pack_stream_(parent.nonblocking_streams_[0]),
      all_reduce_stream_(parent.nonblocking_streams_[1]),
      unpack_stream_(parent.nonblocking_streams_[2]) {
  dtypes dtype = get_dtype<Tc>();

  /* Split gpu_memory into buffers of size n_params_threshold */
  Tc *buff = this->gpu_memory_->cast(dtype, this->parent_.ctx_)
                 ->template pointer<Tc>();
  for (size_t i = 0; i < this->gpu_memory_->size() / this->n_params_threshold_;
       ++i) {
    this->buffers_.emplace(
        buff + i * n_params_threshold,
        SingletonManager::get<Cuda>()->cuda_event(cudaEventDisableTiming));
  }

  this->workspace_ = this->allocate_workspace(this->pack_stream_);
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::AllReduceCallback::
    on_finish_function_backward(const CgFunctionPtr &ptr) {

  /* Find pointers to send (i.e., find pointers that are contained in
   * this->device_ptrs_) */
  vector<std::pair<Tc *, size_t>> device_ptr_list;
  device_ptr_list.reserve(ptr->function_inputs().size());
  for (auto &input : ptr->function_inputs()) {
    if (this->device_ptrs_.find(input->grad()) != this->device_ptrs_.end()) {
      if (parent_.mpi_check_all(input->grad()->array()->zeroing(), "world")) {
        // Skip as the gradient array is not updated.
        continue;
      }
      Tc *device_ptr = input->cast_grad_and_get_pointer<Tc>(this->parent_.ctx_);
      device_ptr_list.push_back(std::make_pair(device_ptr, input->size()));
    }
  }

  if (device_ptr_list.empty()) {
    /* Do nothing because there is no pointers to send */
    return;
  }

  /* Wait until backward computation of this function is completed. */
  auto event =
      SingletonManager::get<Cuda>()->cuda_event(cudaEventDisableTiming);
  NBLA_CUDA_CHECK(cudaEventRecord(*event, nullptr));
  NBLA_CUDA_CHECK(cudaStreamWaitEvent(this->pack_stream_, *event, 0));

  /* Packing phase */
  for (auto &elem : device_ptr_list) {
    Tc *device_ptr = elem.first;
    auto n_param = elem.second;

    while (n_param > 0) {
      /* Pack device_ptr into this->workspace.
       *   If the length of device_ptr exceeds the free space of
       *   this->workspace,
       *   the remained data in device_ptr will be packed into a next workspace.
       */
      auto length =
          std::min<size_t>(n_param, this->n_params_threshold_ -
                                        this->workspace_.n_param_buffered);
      NBLA_CUDA_CHECK(cudaMemcpyAsync(
          this->workspace_.gpu_buffer + this->workspace_.n_param_buffered,
          device_ptr, sizeof(Tc) * length, cudaMemcpyDeviceToDevice,
          this->pack_stream_));
      this->workspace_.n_param_buffered +=
          length; //< Update the length of used space.
      this->workspace_.variables.emplace_back(
          device_ptr, length); //< Store a pointer and size to unpack.

      /* The device_ptr and n_param should refer to the remained data. */
      n_param -= length;
      device_ptr = device_ptr + length;

      if (this->workspace_.n_param_buffered >= this->n_params_threshold_) {
        /* Finish packing because this->workspace_ is full. */
        this->all_reduce(this->workspace_);
        this->unpack(this->workspace_);

        /* Use a next GPU workspace in the packing phase */
        this->release_workspace(this->workspace_, this->unpack_stream_);
        this->workspace_ = this->allocate_workspace(this->pack_stream_);
        /* Notes: workspace_.n_param_buffered is initialized in the above
         *        function. */
      }
    }
  }
}
template <typename T>
void MultiProcessDataParallelCommunicatorNccl<
    T>::AllReduceCallback::on_finish_backward() {
  if (this->workspace_.n_param_buffered != 0) {
    /* Do all_reduce and unpack because the data are remained. */
    this->all_reduce(this->workspace_);
    this->unpack(this->workspace_);
  }

  /* Wait in the default stream until all_reduce is completed */
  auto event =
      SingletonManager::get<Cuda>()->cuda_event(cudaEventDisableTiming);
  NBLA_CUDA_CHECK(cudaEventRecord(*event, this->unpack_stream_));
  NBLA_CUDA_CHECK(cudaStreamWaitEvent(0, *event, 0));
}
template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::AllReduceCallback::all_reduce(
    Workspace &data) {
  /* All-reduce phase */
  /* Wait for the packing phase */
  NBLA_CUDA_CHECK(cudaEventRecord(*data.event, this->pack_stream_));
  NBLA_CUDA_CHECK(
      cudaStreamWaitEvent(this->all_reduce_stream_, *data.event, 0));

  /* All-Reduce */
  this->parent_.all_reduce(data.gpu_buffer, data.n_param_buffered,
                           this->all_reduce_stream_, this->division_, false,
                           this->group_);
}

template <typename T>
void MultiProcessDataParallelCommunicatorNccl<T>::AllReduceCallback::unpack(
    Workspace &data) {
  /* Unpacking phase */
  /* Wait for the all-reduce phase */
  NBLA_CUDA_CHECK(cudaEventRecord(*data.event, this->all_reduce_stream_));
  NBLA_CUDA_CHECK(cudaStreamWaitEvent(this->unpack_stream_, *data.event, 0));

  /* Unpack the packed data into original space */
  int offset = 0;
  for (auto &variable : data.variables) {
    auto device_ptr = variable.first;
    auto n_param = variable.second;

    NBLA_CUDA_CHECK(cudaMemcpyAsync(
        device_ptr, data.gpu_buffer + offset, sizeof(Tc) * n_param,
        cudaMemcpyDeviceToDevice, this->unpack_stream_));

    offset += n_param;
  }
}

template <typename T>
auto MultiProcessDataParallelCommunicatorNccl<
    T>::AllReduceCallback::allocate_workspace(cudaStream_t stream)
    -> Workspace {
  /* Get an unused workspace */
  auto buffer = this->buffers_.front();
  this->buffers_.pop();
  Workspace retval{buffer.first, buffer.second, 0, {}};

  /* Wait until the unpacking phase using this workspace is completed. */
  NBLA_CUDA_CHECK(cudaStreamWaitEvent(stream, *retval.event, 0));
  return retval;
}
template <typename T>
void MultiProcessDataParallelCommunicatorNccl<
    T>::AllReduceCallback::release_workspace(Workspace &workspace,
                                             cudaStream_t stream) {
  /* Notify that this workspace is currently not used (i.e., unpacking phase is
   * completed). */
  NBLA_CUDA_CHECK(cudaEventRecord(*workspace.event, stream));
  /* Store GPU memory into unused memory space set. */
  this->buffers_.emplace(workspace.gpu_buffer, workspace.event);
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
  cuda_device_synchronize(std::to_string(device_id_));
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
    vp->get_grad_pointer<Tc>(ctx);
  }
}

template class MultiProcessDataParallelCommunicatorNccl<float>;
template class MultiProcessDataParallelCommunicatorNccl<Half>;
}
