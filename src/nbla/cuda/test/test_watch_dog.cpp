// Copyright 2020,2021 Sony Corporation.
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

#include "gtest/gtest.h"
#include <chrono>
#include <nbla/cuda/communicator/watch_dog.hpp>
#include <nbla/exception.hpp>
#include <thread>

#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>

namespace nbla {

TEST(WatchDogTest, SimpleConstructWatchdog) { Watchdog watch_dog; }

TEST(WatchDogTest, TestFinishedInTime) {
  Watchdog watch_dog;
  {
    Watchdog::WatchdogLock lck(watch_dog);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
}

TEST(WatchDogTest, TestMultiTimeCheck) {
  Watchdog watch_dog(2);
  for (int i = 0; i < 8; ++i) {
    Watchdog::WatchdogLock lck(watch_dog);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

TEST(WatchDogTest, NotAllowNested) {
  try {
    Watchdog watch_dog(5);
    {
      Watchdog::WatchdogLock lck(watch_dog);
      Watchdog::WatchdogLock lck1(watch_dog);
      FAIL() << "Not raise an exception";
    }
  } catch (const Exception &expected) {
    printf("catched exception!\n");
  }
}

TEST(WatchDogTest, TestLocalTimeoutSetting) {
  Watchdog watch_dog(3);
  {
    Watchdog::WatchdogLock lck(watch_dog, 2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  {
    Watchdog::WatchdogLock lck(watch_dog, 3);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  {
    Watchdog::WatchdogLock lck(watch_dog, 4);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
  }
}

TEST(WatchDogTest, TestDisableWithEnv) {
  setenv("NNABLA_MPI_WATCH_DOG_ENABLE", "0", 0);
  Watchdog watch_dog(3);
  {
    Watchdog::WatchdogLock lck(watch_dog, 1);
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }
  setenv("NNABLA_MPI_WATCH_DOG_ENABLE", "1", 1);
}

TEST(WatchDogTest, TestEnvTimeout) {
  setenv("NNABLA_MPI_WATCH_DOG_TIMEOUT", "3", 0);
  Watchdog watch_dog(1);
  {
    Watchdog::WatchdogLock lck(watch_dog);
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
  }
}

#if 0
// These 2 cases are skipped.
// This one is due to too long testing time.
TEST(WatchDogTest, LongTimeoutTest) {
  Watchdog watch_dog;
  Watchdog::WatchdogLock lck(watch_dog, 150);
  std::this_thread::sleep_for(std::chrono::milliseconds(14900));
}

// This one is that it can cause test stopped with error.
// Since the exception from watchdog thread cannot be catched
// in caller thread.
TEST(WatchDogTest, TestFinishedOutTime) {
  try {
     Watchdog watch_dog(5);
     {
       Watchdog::WatchdogLock lck(watch_dog);
       std::this_thread::sleep_for(std::chrono::milliseconds(700));
       FAIL() << "Not raise an exception";
     }
  }
  catch(const Exception& expected) {
      printf("catched exception!\n");
  }
}
#endif
} // namespace nbla
