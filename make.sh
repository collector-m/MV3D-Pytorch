cd ./net/lib/
python ./setup.py build_ext --inplace
./make.sh
cd ../../

ln -s ./net/lib/roi_pooling_layer/roi_pooling.so ./net/roipooling_op/roi_pooling.so
#ln -s ./net/lib/nms/gpu_nms.cpython-36m-x86_64-linux-gnu.so ./net/processing/gpu_nms.cpython-36m-x86_64-linux-gnu.so
#ln -s ./net/lib/nms/cpu_nms.cpython-36m-x86_64-linux-gnu.so ./net/processing/cpu_nms.cpython-36m-x86_64-linux-gnu.so
#ln -s ./net/lib/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so ./net/processing/cython_bbox.cpython-36m-x86_64-linux-gnu.so

ln -s ./net/lib/nms/gpu_nms.cpython-36m-x86_64-linux-gnu.so ./net/processing/gpu_nms.so
ln -s ./net/lib/nms/cpu_nms.cpython-36m-x86_64-linux-gnu.so ./net/processing/cpu_nms.so
ln -s ./net/lib/utils/cython_bbox.cpython-36m-x86_64-linux-gnu.so ./net/processing/cython_bbox.so
