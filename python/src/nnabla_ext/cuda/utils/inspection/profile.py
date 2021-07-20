# Copyright 2020,2021 Sony Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from nnabla_ext.cuda.init import Event

from collections import OrderedDict
import numpy as np

from nnabla.utils.inspection import FunctionHookCallbackBase


class CudaEventTimerCallback(FunctionHookCallbackBase):
    def __init__(self, ext_name, device_id):
        super(CudaEventTimerCallback, self).__init__()

        self.results = OrderedDict()
        self.device_id = int(device_id)
        self.key_to_events = OrderedDict()

    @property
    def pre_hook(self):
        def callback(key):
            if key not in self.key_to_events:
                self.key_to_events[key] = [Event(self.device_id)]

            self.key_to_events[key][0].record()

        return callback

    @property
    def post_hook(self):
        def callback(key):
            if key not in self.key_to_events:
                raise ValueError("Cannot find pre event for the key ({})".format(key))

            if len(self.key_to_events[key]) == 1:
                self.key_to_events[key].append(Event(self.device_id))

            self.key_to_events[key][1].record()

        return callback

    def calc_elapsed_time(self):
        for key, (start_event, end_event) in self.key_to_events.items():
            end_event.synchronize()
            elapsed = end_event.elapsed(start_event)

            if key not in self.results:
                self.results[key] = []

            self.results[key].append(elapsed)

    def average(self, start_index=0):
        ret = {}
        for key, times in self.results.items():
            ret[key] = {
                "time": np.mean(times[start_index:]),
                "num": len(times) - start_index,
                "scale": "ms"
            }

        return ret
