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

#include <chrono>
#include <cstdlib>
#include <nbla/cuda/communicator/watch_dog.hpp>
#include <nbla/exception.hpp>
#include <stdio.h>
#include <string.h>

namespace nbla {

void Watchdog::watch_dog_loop() {
  std::unique_lock<std::mutex> lck(mutex_);
  {
    std::unique_lock<std::mutex> blck(bootup_);
    bootup_flag_ = 1;
    bcv_.notify_one();
  }
  while (!exit_flag_) {
    if (state_ == START_WATCH_DOG) {
      int32_t timeout_set = TICK * timeout_ticks_;
      if (env_timeout_ > 0) {
        timeout_set = env_timeout_;
      }
      std::cv_status r =
          cv_.wait_for(lck, std::chrono::milliseconds(timeout_set));
      if (r == std::cv_status::timeout) {
        const char *e = std::getenv("NNABLA_MPI_WATCH_DOG_ENABLE");
        if (!e || *e == '0') {
          fprintf(stderr,
                  "WARNING: some node stop response for %8.2f seconds!\n",
                  timeout_set / 1000.0);
          break;
        } else {
          NBLA_ERROR(error_code::runtime,
                     "System stop response within %8.2f seconds!",
                     timeout_set / 1000.0);
        }
      }
    } else {
      cv_.wait(lck);
    }
  }
}

Watchdog::Watchdog(int timeout_ticks)
    : state_(0), exit_flag_(0), timeout_ticks_(timeout_ticks), env_timeout_(-1),
      mutex_(), cv_(), bootup_flag_(0), bootup_(), bcv_(), in_lock_(false),
      thread_(&Watchdog::watch_dog_loop, this) {
  const char *c_t = std::getenv("NNABLA_MPI_WATCH_DOG_TIMEOUT");
  if (nullptr != c_t) {
    int32_t t = std::stoi(c_t);
    if (t > 0)
      env_timeout_ = t * 1000; // user setting is n seconds.
  }

  std::unique_lock<std::mutex> lck(bootup_);
  while (!bootup_flag_)
    bcv_.wait(lck);
}

Watchdog::~Watchdog() {
  {
    std::unique_lock<std::mutex> lck(mutex_);
    exit_flag_ = 1;
    state_ = STOP_WATCH_DOG;
    cv_.notify_one();
  }
  thread_.join();
}

Watchdog::WatchdogLock::WatchdogLock(Watchdog &wd, int timeout_ticks)
    : wd_(wd), previous_timeout_(-1) {
  if (wd_.in_lock_) {
    NBLA_ERROR(error_code::value, "Watchdog lock nested is not allowed.");
  }
  wd_.in_lock_ = true;
  std::unique_lock<std::mutex> lck(wd_.mutex_);
  if (timeout_ticks > 0) {
    previous_timeout_ = wd_.timeout_ticks_;
    wd_.timeout_ticks_ = timeout_ticks;
  }
  wd_.state_ = START_WATCH_DOG;
  wd_.cv_.notify_all();
}

Watchdog::WatchdogLock::~WatchdogLock() {
  std::unique_lock<std::mutex> lck(wd_.mutex_);
  wd_.state_ = STOP_WATCH_DOG;
  if (previous_timeout_ != -1)
    wd_.timeout_ticks_ = previous_timeout_;
  wd_.cv_.notify_all();
  wd_.in_lock_ = false;
}
} // namespace nbla
