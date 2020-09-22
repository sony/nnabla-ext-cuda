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

/** Communicator interface class
 */
#ifndef __NBLA_NCCL_MULTIPROCESS_DATAPARALLELCOMMUNICATOR_HPP__
#define __NBLA_NCCL_MULTIPROCESS_DATAPARALLELCOMMUNICATOR_HPP__
#include <nbla/array.hpp>
#include <nbla/communicator/multi_process_data_parallel_communicator.hpp>
#include <nbla/computation_graph/function.hpp>
#include <nbla/context.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/cuda/communicator/nccl_utils.hpp>
#include <nbla/cuda/communicator/watch_dog.hpp>
#include <nbla/variable.hpp>

#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "nccl.h"

namespace nbla {

using std::string;
using std::vector;
using std::shared_ptr;
using std::unordered_map;
using std::unordered_set;
using std::pair;

/** Forward decl. of a wrapper object of MPI_Comm
 */
class MpiCommWrapper;

/** \addtogroup NNablaCoreGrp */
/*@{*/

/** Communicator interface which is extended to implement a new Communicator
class.

Communicator exchanges gradients parameters or parameters itself.

*/

template <typename T>
class NBLA_API MultiProcessDataParallelCommunicatorNccl
    : public MultiProcessDataParallelCommunicator<T> {
protected:
  int all_reduce_timeout_ = 500; // timeout=50s
  Watchdog watch_dog_;
  int device_id_;

  static bool mpi_initialized_;

  // Device streams initialized in init method
  cudaStream_t stream_;
  int num_streams_ = 10; // TODO: hard-codded.
  vector<cudaStream_t> streams_ = vector<cudaStream_t>(num_streams_);
  /* These streams are used in all_reduce_callback.
   *   nonblocking_streams_[0]: the stream used to pack data into a large
   *                            buffer.
   *   nonblocking_streams_[1]: the stream used to all-reduce data.
   *   nonblocking_streams_[2]: the stream used to unpack data from a large
   *                            buffer.
   */
  vector<cudaStream_t> nonblocking_streams_ = vector<cudaStream_t>(3);

  // Groups
  unordered_map<string, ncclComm_t> comms_;
  unordered_map<string, shared_ptr<MpiCommWrapper>> mpi_comms_;

  bool mpi_check_any(bool condition, const string &group);
  bool mpi_check_all(bool condition, const string &group);

  vector<NdArrayPtr> get_modified_arrays(const vector<NdArrayPtr> &arrays,
                                         const string &group);

public:
  typedef typename CudaType<T>::type Tc;
  MultiProcessDataParallelCommunicatorNccl(const Context &ctx);
  virtual ~MultiProcessDataParallelCommunicatorNccl();
  virtual string name() { return "MultiProcessDataParallelCommunicatorNccl"; }

  /** Adding context and parameters communicated via this class.
  @param cparams pair<Context, vector of pair<name, VariablePtr>
  */
  void add_context_and_parameters(
      const pair<Context, vector<pair<string, VariablePtr>>> &ctx_params);

  /** Remove previously registered parameters by keys.
   */
  void remove_context_parameters(const pair<Context, vector<string>> &ctx_keys);

  /** Clear all parameters.
   */
  void clear_context_parameters();

  /** Initall or initrank, depending multi-threads or multi-processes.
   * This function MUST be called after all parameters communicated
   * are added by `add_context_and_parameters` method.
  */
  virtual void init();

  virtual void barrier();

  virtual void abort();

  virtual string new_group(pair<string, vector<int>> name_ranks_pair);

  virtual shared_ptr<NdArray>
  copy_inside_device(const vector<NdArrayPtr> &ndarray_list);
  virtual void copy_back_inside_device(const vector<NdArrayPtr> &ndarray_list,
                                       NdArrayPtr large_ndarray);

  virtual void reduce(const vector<NdArrayPtr> &ndarray_list, int dst,
                      bool division = false, bool inplace = false,
                      const string &group = "world");
  virtual void reduce(NdArrayPtr ndarray, int dst, bool division = false,
                      bool inplace = false, const string &group = "world");
  virtual void reduce(NdArrayPtr ndarray, cudaStream_t stream, int dst,
                      bool division = false, bool inplace = false,
                      const string &group = "world");
  virtual void allreduce(bool division = false, bool inplace = false);
  virtual void all_reduce(const vector<NdArrayPtr> &ndarray_list,
                          bool division = false, bool inplace = false,
                          const string &group = "world");
  virtual void all_reduce(NdArrayPtr ndarray, bool division = false,
                          bool inplace = false, const string &group = "world");
  virtual void all_reduce(NdArrayPtr ndarray, cudaStream_t stream,
                          bool division = false, bool inplace = false,
                          const string &group = "world");
  virtual void all_reduce(Tc *gpu_buffer, size_t n_param, cudaStream_t stream,
                          bool division = false, bool inplace = false,
                          const string &group = "world");
  virtual CommunicatorBackwardCallbackPtr
  all_reduce_callback(const vector<NdArrayPtr> &ndarray_list, size_t pack_size,
                      bool division = false, const string &group = "world");
  virtual CommunicatorBackwardCallbackPtr
  all_reduce_callback(NdArrayPtr ndarray, size_t pack_size,
                      bool division = false, const string &group = "world");
  virtual void reduce_scatter(const vector<NdArrayPtr> &ndarray_list,
                              NdArrayPtr ndarray, bool division = false,
                              const string &group = "world");
  virtual void bcast(const vector<NdArrayPtr> &ndarray_list, int src,
                     bool inplace = false, const string &group = "world");
  virtual void bcast(NdArrayPtr ndarray, int src, bool inplace = false,
                     const string &group = "world");
  virtual void bcast(NdArrayPtr ndarray, cudaStream_t stream, int src,
                     bool inplace = false, const string &group = "world");
  virtual void all_gather(NdArrayPtr ndarray,
                          const vector<NdArrayPtr> &ndarray_list,
                          const string &group = "world");

  virtual void reduce_async(bool division = false);
  virtual void allreduce_async(bool division = false, bool inplace = false);
  virtual void reducescatter_async(bool division = false);
  virtual void bcast_async();
  virtual void allgather_async();

  /** Sync all parameters added in this communicator based on `Context`.
   * Coerce to copy all parameters to the device specified by `Context`.
   *
   */
  void sync_all_params();

  /** Get array classes that are allowed to be specified by Context
  */
  vector<string> allowed_array_classes();

protected:
  const static int max_gpu_memory_size =
      1024 * 1024 * 40; // TODO: magic-number (40MiB)

  class AllReduceCallback : public CommunicatorBackwardCallback {
  public:
    AllReduceCallback(MultiProcessDataParallelCommunicatorNccl<T> &parent,
                      const string &group, size_t n_params_threshold,
                      bool division, const NdArrayPtr &gpu_memory,
                      const unordered_set<NdArrayPtr> &device_ptrs);

    virtual void on_finish_function_backward(const CgFunctionPtr &ptr) override;
    virtual void on_finish_backward() override;

  private:
    /** GPU memory space for packing and unpacking */
    struct Workspace {
      Tc *gpu_buffer;
      shared_ptr<cudaEvent_t> event;
      size_t n_param_buffered;
      vector<pair<Tc *, size_t>> variables;
    };
    using Buffer = std::pair<Tc *, shared_ptr<cudaEvent_t>>;

    void all_reduce(Workspace &data);
    void unpack(Workspace &data);

    Workspace allocate_workspace(cudaStream_t stream);
    void release_workspace(Workspace &workspace, cudaStream_t stream);

    MultiProcessDataParallelCommunicatorNccl<T> &parent_;

    const string group_;

    const size_t n_params_threshold_;
    const bool division_;
    const unordered_set<NdArrayPtr> device_ptrs_; //< GPU pointers to send.

    const NdArrayPtr gpu_memory_;
    std::queue<Buffer>
        buffers_; //< GPU memory spaces that are not used currently.

    Workspace workspace_; //< The workspace used in the packing phase.

    cudaStream_t pack_stream_;
    cudaStream_t all_reduce_stream_;
    cudaStream_t unpack_stream_;
  };

  void wait_by_device_synchronization();
  void wait_by_streams_synchronization();
  void divide_by_num_devices(bool division);

  DISABLE_COPY_AND_ASSIGN(MultiProcessDataParallelCommunicatorNccl);
};
}
#endif
