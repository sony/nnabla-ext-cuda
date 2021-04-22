// Copyright 2019,2020,2021 Sony Corporation.
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

#include "nbla/cuda/nvtx.hpp"
#ifndef _WIN32
#include <dlfcn.h>
#include <nvToolsExt.h>
#endif
#include <string>
#include <vector>

namespace nbla {
#ifndef _WIN32
int (*nvtxInitialize)(const nvtxInitializationAttributes_t *) = nullptr;
void (*nvtxMarkA)(const char *) = nullptr;
int (*nvtxRangePushA)(const char *) = nullptr;
int (*nvtxRangePushEx)(const nvtxEventAttributes_t *) = nullptr;
int (*nvtxRangePop)(void) = nullptr;

static const char *NVTX_DYNAMIC_LIBRARY = "libnvToolsExt.so";
static void *nvtx_library_handle = nullptr;

int dl_nvtx_init(void) {
  dlerror(); // Clear any existing error

  nvtx_library_handle = dlopen(NVTX_DYNAMIC_LIBRARY, RTLD_NOW);
  if (nvtx_library_handle == nullptr) {
    return -1;
  }

  nvtxInitialize = (int (*)(const nvtxInitializationAttributes_t *))dlsym(
      nvtx_library_handle, "nvtxInitialize");
  nvtxMarkA = (void (*)(const char *))dlsym(nvtx_library_handle, "nvtxMarkA");
  nvtxRangePushA =
      (int (*)(const char *))dlsym(nvtx_library_handle, "nvtxRangePushA");
  nvtxRangePushEx = (int (*)(const nvtxEventAttributes_t *))dlsym(
      nvtx_library_handle, "nvtxRangePushEx");
  nvtxRangePop = (int (*)(void))dlsym(nvtx_library_handle, "nvtxRangePop");

  const char *error = dlerror();
  if (error != nullptr) {
    fprintf(stderr, "%s\n", error);
    dl_nvtx_finish();
    return -1;
  }
  return 0;
}

int dl_nvtx_finish(void) {
  if (nvtx_library_handle != nullptr) {
    nvtxInitialize = nullptr;
    nvtxMarkA = nullptr;
    nvtxRangePushA = nullptr;
    nvtxRangePushEx = nullptr;
    nvtxRangePop = nullptr;
    dlclose(nvtx_library_handle);
    nvtx_library_handle = nullptr;
  }
  return 0;
}

using std::string;
using std::vector;

const vector<uint32_t> colors = {
    0x00ff0000, // red
    0x0000ff00, // green
    0x000000ff, // blue
    0x00dc143c, // crimson
    0x00006400, // darkgreen
    0x0000ffff, // cyan
    0x00ffa500, // orange
    0x008b008b, // darkmagenta
    0x00000080  // navy
};
static int color_id = 0;

inline void nvtxRangePushWithColorChange(const char *msg) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = colors[color_id++];
  color_id %= colors.size();
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.message.ascii = msg;

  if (nvtxRangePushEx != nullptr) {
    nvtxRangePushEx(&eventAttrib);
  }
}

////////////////////////////////////////
// Define functions used by cython
////////////////////////////////////////
void nvtx_mark_A(string msg) {
  if (nvtxMarkA != nullptr) {
    nvtxMarkA(msg.c_str());
  }
}

void nvtx_range_push_A(string msg) {
  if (nvtxRangePushA != nullptr) {
    nvtxRangePushA(msg.c_str());
  }
}

void nvtx_range_push_with_C(string msg) {
  nvtxRangePushWithColorChange(msg.c_str());
}

void nvtx_range_pop() {
  if (nvtxRangePop != nullptr) {
    nvtxRangePop();
  }
}
#else
int dl_nvtx_init(void) { return 0; }
int dl_nvtx_finish(void) { return 0; }
void nvtx_mark_A(string msg) {}
void nvtx_range_push_A(string msg) {}
void nvtx_range_push_with_C(string msg) {}
void nvtx_range_pop() {}
#endif
} // namespace nbla
