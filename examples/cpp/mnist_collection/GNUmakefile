# Copyright 2019,2020,2021 Sony Corporation.
# Copyright 2021,2022 Sony Group Corporation.
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
all: train_lenet_classifier_cuda.cpp train_resnet_classifier_cuda.cpp train_vae_cuda.cpp train_dcgan_cuda.cpp train_siamese_cuda.cpp train_vat_cuda.cpp
	$(CXX) -std=c++14 -O -o train_lenet_classifier_cuda train_lenet_classifier_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX) -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/
	$(CXX) -std=c++14 -O -o train_resnet_classifier_cuda train_resnet_classifier_cuda.cpp -lnnabla_utils -lnnabla -lnnabla_cuda$(CUDA_VERSION_SUFFIX) -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/
	$(CXX) -std=c++14 -O -o train_vae_cuda train_vae_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/
	$(CXX) -std=c++14 -O -o train_dcgan_cuda train_dcgan_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/
	$(CXX) -std=c++14 -O -o train_siamese_cuda train_siamese_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/
	$(CXX) -std=c++14 -O -o train_vat_cuda train_vat_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/

lenet: train_lenet_classifier_cuda.cpp
	$(CXX) -std=c++14 -O -o train_lenet_classifier_cuda train_lenet_classifier_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/

resnet: train_resnet_classifier_cuda.cpp
	$(CXX) -std=c++14 -O -o train_resnet_classifier_cuda train_resnet_classifier_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/

vae: train_vae_cuda.cpp
	$(CXX) -std=c++14 -O -o train_vae_cuda train_vae_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/

dcgan: train_dcgan_cuda.cpp
	$(CXX) -std=c++14 -O -o train_dcgan_cuda train_dcgan_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/

siamese: train_siamese_cuda.cpp
	$(CXX) -std=c++14 -O -o train_siamese_cuda train_siamese_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/

vat: train_vat_cuda.cpp
	$(CXX) -std=c++14 -O -o train_vat_cuda train_vat_cuda.cpp -lnnabla -lnnabla_utils -lnnabla_cuda$(CUDA_VERSION_SUFFIX)  -lz -I ${NNABLA_DIR}/examples/cpp/mnist_collection/

clean:
	rm -f train_lenet_classifier_cuda
	rm -f train_resnet_classfier_cuda
	rm -f train_vae_cuda
	rm -f train_dcgan_cuda
	rm -f train_siamese_cuda
	rm -f train_vat_cuda
