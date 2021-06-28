/*
 * nvcc -ccbin g++ -m64 cudalibCheck.cpp -o cudalibCheck -lcuda
 *
 */
#include <stdio.h>
#include "cuda.h"


// Host code
int main(int argc, char **argv) {
  CUresult ret;
  bool usable = true;
  int version = -1;
  int count = -1;
  
  ret = cuInit(0);
  if (ret == CUDA_SUCCESS) {
    ret = cuDriverGetVersion(&version);
    if (ret != CUDA_SUCCESS) {
      fprintf(stderr, "fail to get version: %d\n", ret);
      usable = false;
    }

    ret = cuDeviceGetCount(&count);
    if (ret != CUDA_SUCCESS) {
      fprintf(stderr, "fail to get device count: %d\n", ret);
      usable = false;
    }

    for (int i=0; i<count; i++) {
      CUdevice device;
      ret = cuDeviceGet(&device, i);
      if (ret != CUDA_SUCCESS) {
	usable = false;
	break;
      }
    }
  }
  else {
    usable = false;
  }
  
  if (usable) {
    int major = version / 1000;
    int minor = (version - major * 1000) / 10;
    if ((11000 <= version) && (0 < count)) {
      printf("CUDA Driver OK. Version:%d.%d/Device:%d\n", major, minor, count);
      return 0;
    }
    else {
      printf("Insufficient driver. Version:%d.%d/Device:%d\n", major, minor, count);
      return 1;
    }
  }
  else {
    printf("Insufficient driver. %d\n", ret);
    return 2;
  }
}
