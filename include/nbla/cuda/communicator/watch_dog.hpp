// Copyright (c) 2020 Sony Corporation. All Rights Reserved.
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
#include <condition_variable>
#include <mutex>
#include <thread>

namespace nbla {

class Watchdog {
private:
  static const int TIMEOUT_TICKS = 600;    // default timeout is 60s
  static const int TICK = 100;             // 100 ms each tick
  static const int STOP_WATCH_DOG = -1000; // arbitrary negative value
  static const int EXIT_WATCH_DOG = 1;     // exit flag
  static const int START_WATCH_DOG = 1;    // a positive value to start
  int state_;
  int exit_flag_;
  int timeout_ticks_;
  std::mutex mutex_;
  std::condition_variable cv_;
  int bootup_flag_;
  std::mutex bootup_;
  std::condition_variable bcv_;
  bool in_lock_;
  std::thread thread_;
  void watch_dog_loop();

public:
  Watchdog(int timeout_ticks = TIMEOUT_TICKS);
  ~Watchdog();
  friend class WatchdogLock;
  class WatchdogLock {
  public:
    WatchdogLock(Watchdog &wd, int timeout_ticks = -1);
    ~WatchdogLock();

  private:
    Watchdog &wd_;
    int previous_timeout_;
  };
};
}
