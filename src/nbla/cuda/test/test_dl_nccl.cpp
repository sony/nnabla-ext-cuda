// Copyright 2022 Sony Group Corporation.
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
#include <nbla/cuda/communicator/dl_nccl.h>

TEST(DlNcclTest, TestInitialize) {
  ASSERT_EQ(0, dl_nccl_init());
  ncclComm_t comm;
  ncclUniqueId comm_id;
  ncclGetUniqueId(&comm_id);
  ASSERT_EQ(ncclSuccess, ncclCommInitRank(&comm, 1, comm_id, 0));
  ASSERT_EQ(ncclSuccess, ncclCommDestroy(comm));
}
