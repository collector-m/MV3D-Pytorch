#!/usr/bin/env bash
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
echo $TF_INC

CUDA_PATH=/usr/local/cuda-8.0/
Python_path = /usr/include/


cd roi_pooling_layer

nvcc -std=c++11 -c -o roi_pooling_op.cu.o roi_pooling_op_gpu.cu.cc \
		-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CXXFLAGS \
		-arch=sm_30

## if you install tf using already-built binary, or gcc version 4.x, uncomment the two lines below
#g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o roi_pooling.so roi_pooling_op.cc \
#	roi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64

# for gcc5-built tf
g++ -std=c++11 -shared -o roi_pooling.so roi_pooling_op.cc \
	roi_pooling_op.cu.o -I $TF_INC  -D GOOGLE_CUDA=1 -fPIC $CXXFLAGS  -D_GLIBCXX_USE_CXX11_ABI=0 \
	-lcudart -L $CUDA_PATH/lib64 -D_GLIBCXX_USE_CXX11_ABI=0 
cd ..


# add building psroi_pooling layer
#cd psroi_pooling_layer
#nvcc -std=c++11 -c -o psroi_pooling_op.cu.o psroi_pooling_op_gpu.cu.cc \
#	-I $TF_INC -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_52
#
#g++ -std=c++11 -shared -o psroi_pooling.so psroi_pooling_op.cc \
#	psroi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64
#
## if you install tf using already-built binary, or gcc version 4.x, uncomment the two lines below
#g++ -std=c++11 -shared -D_GLIBCXX_USE_CXX11_ABI=0 -o psroi_pooling.so psroi_pooling_op.cc \
#	psroi_pooling_op.cu.o -I $TF_INC -fPIC -lcudart -L $CUDA_PATH/lib64

#cd ..


