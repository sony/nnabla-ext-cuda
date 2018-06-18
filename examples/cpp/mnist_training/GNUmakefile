all: main.cpp
	$(CXX) -std=c++11 -O -o mnist_training_cuda main.cpp -lnnabla -lnnabla_utils -lnnabla_cuda -lz -I ${NNABLA_DIR}/examples/cpp/mnist_training/

clean:
	rm -f mnist_training
