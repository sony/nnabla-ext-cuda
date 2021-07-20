# Copyright 2019,2020,2021 Sony Corporation.
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


enabled = False
try:
    import nvidia.dali
    from ._dali_iterator import *
    enabled = True
except ImportError as e:
    print(e)
    from nnabla.logger import logger
    logger.error("Skip importing nnabla daliIterator wrapper because nvidia.dali is not found."
                 " Please make sure you've installed nvidia.dali.")
