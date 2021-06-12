// Copyright 2018,2019,2020,2021 Sony Corporation.
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

#include <fstream>
#include <iostream>
#include <nbla/context.hpp>
#include <nbla/cuda/cudnn/init.hpp>
#include <nbla/cuda/init.hpp>
#include <nbla_cli/nbla_train.hpp>

static void print_usage_and_exit(const char *name) {
  std::cerr << "Usage: " << name << " (train)" << std::endl;
  std::cerr << "    " << name << " train input_files ..." << std::endl;
  std::cerr << "               input_file must be nnp." << std::endl;
  exit(-1);
}

#ifndef _WIN32
bool nbla_cuda_train(int argc, char *argv[]) {

  // Create a context (the following setting is recommended.)
  nbla::init_cudnn();
  nbla::Context ctx{
      {"cudnn:float", "cuda:float", "cpu:float"}, "CudaCachedArray", "0"};

  if (!nbla_train_core(ctx, argc, argv)) {
    return false;
  }

  return true;
}
#else
bool nbla_cuda_train(int argc, char *argv[]) {
  std::cerr << "Error: nbla_cuda train is not supported for windows."
            << std::endl;
  return false;
}
#endif

int main(int argc, char *argv[]) {
  const char *command_name = argv[0];

  if (argc < 2) {
    print_usage_and_exit(command_name);
  }

  std::string command(*++argv);
  argc--;

  if (command == "train") {
    nbla_cuda_train(argc, argv);
  } else {
    print_usage_and_exit(command_name);
  }

  return 0;
}
