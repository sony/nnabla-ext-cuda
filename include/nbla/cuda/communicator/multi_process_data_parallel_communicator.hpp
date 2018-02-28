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
#include <nbla/context.hpp>
#include <nbla/cuda/array/cuda_array.hpp>
#include <nbla/variable.hpp>

#include <memory>
#include <string>
#include <unordered_map>

#include "nccl.h"

namespace nbla {

using std::string;
using std::vector;
using std::shared_ptr;
using std::unordered_map;
using std::pair;

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
  int device_id_;

  static bool mpi_initialized_;

  // Device streams initialized in init method
  cudaStream_t stream_;
  int num_streams_ = 10; // TODO: hard-codded.
  vector<cudaStream_t> streams_ = vector<cudaStream_t>(num_streams_);

  // Groups
  unordered_map<string, ncclComm_t> comms_;

public:
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
  void wait_by_device_synchronization();
  void wait_by_streams_synchronization();
  void divide_by_num_divices(bool division);

  DISABLE_COPY_AND_ASSIGN(MultiProcessDataParallelCommunicatorNccl);
};
}
#endif
