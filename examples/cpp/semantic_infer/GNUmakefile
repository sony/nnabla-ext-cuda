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

all: semantic.cpp
	$(CXX) -std=c++14 -D TIMING -O -o semantic semantic.cpp -lnnabla -lnnabla_utils -lcurl -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -I../../../../nnabla/include -I/usr/local/include/opencv4
#	 -L~/.local/lib/python3.9/site-packages/nnabla/bin -L~/.local/lib/python3.9/site-packages/nnabla_ext/cuda
cuda: semantic.cpp
	$(CXX) -std=c++14 -D TIMING -D WITH_CUDA -O -o semantic_cuda semantic.cpp -lnnabla -lnnabla_utils -lnnabla_cuda -lcurl -lopencv_core -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -I../../../include -I../../../../nnabla/include -I/usr/local/include/opencv4
#    -L~/.local/lib/python3.9/site-packages/nnabla/bin -L~/.local/lib/python3.9/site-packages/nnabla_ext/cuda

clean:
	rm -f semantic
