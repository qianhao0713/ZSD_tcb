export LD_LIBRARY_PATH=/home/qianhao/miniconda3/envs/trt/lib/:/home/qianhao/miniconda3/envs/trt/lib/python3.8/site-packages/torch/lib/:/usr/local/cuda-11.3/targets/x86_64-linux/lib/:/usr/local/cuda-11.3/targets/x86_64-linux/lib/:/home/qianhao/TensorRT/lib/:/raid/qianhao/tensorrt/TensorRT-8.6.1.6/lib/:${LD_LIBRARY_PATH}
# ./tools/dist_test.sh configs/zsd/65_15/test/zsd/zsd_TCB_test.py pth/COCO_65_15.pth 4 --json_out results/zsd_65_15.json
# python tools/onnx2trt.py onnx/COCO_65_15.onnx trt/COCO_65_15.trt
python tools/onnx2trt.py onnx/cattle.onnx trt/cattle.trt
