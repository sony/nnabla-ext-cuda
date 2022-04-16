# Copyright 2022 Sony Group Corporation.
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

import nnabla_ext.cuda.nvtx as nvtx


def main():
    nvtx.range_push('nnabla-push-pop-test')
    print('test for push and pop')
    nvtx.range_pop()

    with nvtx.Range('nnabla-range-test'):
        print('test for Range')


if __name__ == '__main__':
    main()
