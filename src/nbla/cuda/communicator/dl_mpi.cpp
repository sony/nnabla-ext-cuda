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
#define DL_MPI_MAIN
#include <cstddef>
#include <dlfcn.h>
#include <nbla/cuda/communicator/dl_mpi.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

const char *MPI_SO_1 = "/usr/lib64/libmpi.so";
const char *MPI_SO_2 = "/usr/lib/libmpi.so";
const char *ENV_HOME_LIST[] = {"OMPI_HOME"};

const char *MPI_SO_SEARCH_LIST[] = {
    "/usr/lib64/libmpi.so", "/usr/lib/libmpi.so", "/opt/openmpi/lib/libmpi.so"};

static void *mpi_library_handle = 0;

int dl_mpi_init(void) {
  char *error;
  void *handle;
  int errcode = -1;

  dlerror(); // Clear any existing error

  for (int i = 0; i < sizeof(ENV_HOME_LIST) / sizeof(char *); ++i) {
    const char *mpi_home = getenv(ENV_HOME_LIST[i]);
    if (mpi_home) {
      char env_so_path[1024];
      snprintf(env_so_path, sizeof(env_so_path), "%s/lib/libmpi.so", mpi_home);
      mpi_library_handle = dlopen(env_so_path, RTLD_LAZY);
      if (mpi_library_handle) {
        errcode = 0;
        break;
      }
    }
  }

  if (errcode < 0) {
    for (int i = 0; i < sizeof(MPI_SO_SEARCH_LIST) / sizeof(char *); ++i) {
      const char *so_path = MPI_SO_SEARCH_LIST[i];
      mpi_library_handle = dlopen(so_path, RTLD_NOW);
      if (mpi_library_handle) {
        errcode = 0;
        break;
      }
    }
  }

  if (errcode < 0) {
    fprintf(stderr, "openmpi library is not found, distribution training "
                    "feature is disabled.\n");
    return -1;
  }

  handle = mpi_library_handle;

  MPI_Init = (int (*)(int *, char ***))dlsym(handle, "MPI_Init");
  MPI_Init_thread =
      (int (*)(int *, char ***, int, int *))dlsym(handle, "MPI_Init_thread");
  MPI_Initialized = (int (*)(int *))dlsym(handle, "MPI_Initialized");
  MPI_Finalized = (int (*)(int *))dlsym(handle, "MPI_Finalized");
  MPI_Error_string =
      (int (*)(int, char *, int *))dlsym(handle, "MPI_Error_string");
  MPI_Group_free = (int (*)(MPI_Group *))dlsym(handle, "MPI_Group_free");
  MPI_Finalize = (int (*)(void))dlsym(handle, "MPI_Finalize");
  MPI_Comm_group =
      (int (*)(MPI_Comm, MPI_Group *))dlsym(handle, "MPI_Comm_group");
  MPI_Group_incl = (int (*)(MPI_Group, int, int *, MPI_Group *))dlsym(
      handle, "MPI_Group_incl");
  MPI_Comm_size = (int (*)(MPI_Comm, int *))dlsym(handle, "MPI_Comm_size");
  MPI_Comm_rank = (int (*)(MPI_Comm, int *))dlsym(handle, "MPI_Comm_rank");
  MPI_Allgather = (int (*)(void *, int, MPI_Datatype, void *, int, MPI_Datatype,
                           MPI_Comm))dlsym(handle, "MPI_Allgather");
  MPI_Barrier = (int (*)(MPI_Comm))dlsym(handle, "MPI_Barrier");
  MPI_Abort = (int (*)(MPI_Comm, int))dlsym(handle, "MPI_Abort");
  MPI_Allreduce = (MPI_Allreduce_t)dlsym(handle, "MPI_Allreduce");
  MPI_Bcast = (MPI_Bcast_t)dlsym(handle, "MPI_Bcast");
  MPI_Comm_create_group =
      (MPI_Comm_create_group_t)dlsym(handle, "MPI_Comm_create_group");

  struct {
    struct ompi_predefined_datatype_t **global;
    const char *name;
  } global_variables[] = {{&_ompi_mpi_c_bool, "ompi_mpi_c_bool"},
                          {&_ompi_mpi_datatype_null, "ompi_mpi_datatype_null"},
                          {&_ompi_mpi_byte, "ompi_mpi_byte"},
                          {&_ompi_mpi_comm_world, "ompi_mpi_comm_world"},
                          {&_ompi_mpi_comm_self, "ompi_mpi_comm_self"},
                          {&_ompi_mpi_comm_null, "ompi_mpi_comm_null"},
                          {&_ompi_mpi_group_empty, "ompi_mpi_group_empty"},
                          {&_ompi_mpi_group_null, "ompi_mpi_group_null"},
                          {&_ompi_message_no_proc, "ompi_message_no_proc"},
                          {&_ompi_message_null, "ompi_message_null"},
                          {&_ompi_mpi_op_null, "ompi_mpi_op_null"},
                          {&_ompi_mpi_op_min, "ompi_mpi_op_min"},
                          {&_ompi_mpi_op_max, "ompi_mpi_op_max"},
                          {&_ompi_mpi_op_sum, "ompi_mpi_op_sum"},
                          {&_ompi_mpi_op_prod, "ompi_mpi_op_prod"},
                          {&_ompi_mpi_op_land, "ompi_mpi_op_land"},
                          {&_ompi_mpi_op_band, "ompi_mpi_op_band"},
                          {&_ompi_mpi_op_lor, "ompi_mpi_op_lor"},
                          {&_ompi_mpi_op_bor, "ompi_mpi_op_bor"},
                          {&_ompi_mpi_op_lxor, "ompi_mpi_op_lxor"},
                          {&_ompi_mpi_op_bxor, "ompi_mpi_op_bxor"},
                          {&_ompi_mpi_op_maxloc, "ompi_mpi_op_maxloc"},
                          {&_ompi_mpi_op_minloc, "ompi_mpi_op_minloc"},
                          {&_ompi_mpi_op_replace, "ompi_mpi_op_replace"},
                          {&_ompi_mpi_op_no_op, "ompi_mpi_op_no_op"}};

  for (int i = 0; i < sizeof(global_variables) / sizeof(global_variables[0]);
       ++i) {
    *global_variables[i].global =
        (ompi_predefined_datatype_t *)dlsym(handle, global_variables[i].name);
  }

  error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "%s\n", error);
    return -1;
  }
  return 0;
}

int dl_mpi_fini(void) {
  if (mpi_library_handle != 0)
    dlclose(mpi_library_handle);
  return 0;
}
}
