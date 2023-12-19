export PYTHONPATH=$PYTHONPATH:/home/qianhao/ZSD_tcb
export LD_LIBRARY_PATH=/home/qianhao/miniconda3/envs/trt/lib/:/home/qianhao/miniconda3/envs/trt/lib/python3.8/site-packages/torch/lib/:/home/qianhao/TensorRT/lib/:/raid/qianhao/tensorrt/TensorRT-8.6.1.6/lib/:${LD_LIBRARY_PATH}
# ./tools/dist_test.sh configs/zsd/65_15/test/zsd/zsd_TCB_test.py pth/COCO_65_15.pth 4 --json_out results/zsd_65_15.json

# python tools/trt_test.py configs/zsd/65_15/test/zsd/zsd_TCB_test.py trt/COCO_65_15.trt --out results/zsd_65_15.pkl
python tools/trt_test.py configs/cattle_config/only_cattle_stone_test.py trt/cattle.trt --out results/cattle.pkl
