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

#include <string>

using std::string;

////////////////////////////////////////
// Define internal use only functions
////////////////////////////////////////

#ifdef WITH_NVTX
#include <nvToolsExt.h>
#include <vector>

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

  nvtxRangePushEx(&eventAttrib);
}

#else

inline void nvtxMarkA(const char *) {}

//
// inline void nvtxMarkEx(...) {}
//
// inline int nvtxRangeStartA(const char *) { return 0; }
//
// inline int nvtxRangeStartEx(...) { return 0; }
//
// inline void nvtxRangeEnd(nvtxRangeId_t) {}
//
inline void nvtxRangePushA(const char *) {}
//
// inline void nvtxRangePushEx(...) {}
//

inline void nvtxRangePushWithColorChange(const char *) {}

inline void nvtxRangePop() {}

#endif // WITH_NVTX

////////////////////////////////////////
// Define functions used by cython
////////////////////////////////////////

namespace nbla {
void nvtx_mark_A(string msg) { nvtxMarkA(msg.c_str()); };

void nvtx_range_push_A(string msg) { nvtxRangePushA(msg.c_str()); };

void nvtx_range_push_with_C(string msg) {
  nvtxRangePushWithColorChange(msg.c_str());
}

void nvtx_range_pop() { nvtxRangePop(); };
}
