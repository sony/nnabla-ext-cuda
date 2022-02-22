// Copyright 2021 Sony Corporation.
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
#if 0
#include "gtest/gtest.h"
#include <nbla/cuda/communicator/dl_mpi.h>

TEST(DlMpiTest, TestInitialize) {
  int flag = 0;
  int rank = 0;
  int argc = 0;
  char **argv = nullptr;
  int requiredThreadLevelSupport = MPI_THREAD_SERIALIZED;
  int provided;
  ASSERT_EQ(0, dl_mpi_init());
  ASSERT_EQ(0, MPI_Initialized(&flag));
  ASSERT_EQ(0, MPI_Init_thread(&argc, &argv, requiredThreadLevelSupport, &provided));
  //ASSERT_EQ(0, MPI_Init(&argc, &argv));
  ASSERT_EQ(0, MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  ASSERT_EQ(0, MPI_Finalize()); 
}
#endif

#include <gtest/gtest.h>
#include <mpi.h>

TEST(DlMpiTest, original) {
  int flag = 0;
  int rank = 0;
  int argc = 0;
  char **argv = nullptr;
  int r = MPI_THREAD_SERIALIZED;
  int p = 0;

  printf("test mpi api...\n");
  ASSERT_EQ(0, MPI_Initialized(&flag));
  ASSERT_EQ(0, MPI_Init_thread(&argc, &argv, r, &p));
  ASSERT_EQ(0, MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  ASSERT_EQ(0, MPI_Finalize());
  printf("rank=%d\n", rank);
}
