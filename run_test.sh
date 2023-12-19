export PYTHONPATH=$PYTHONPATH:/home/qianhao/ZSD_tcb
export LD_LIBRARY_PATH=/home/qianhao/miniconda3/envs/trt/lib/:/home/qianhao/miniconda3/envs/trt/lib/python3.8/site-packages/torch/lib/:/usr/local/cuda-11.4/targets/x86_64-linux/lib/:/usr/local/cuda-11.4/targets/x86_64-linux/lib/:/raid/qianhao/pkgs/cudnn/lib/:${LD_LIBRARY_PATH}
# ./tools/dist_test.sh configs/zsd/65_15/test/zsd/zsd_TCB_test.py pth/COCO_65_15.pth 4 --json_out results/zsd_65_15.json
# python tools/test.py configs/zsd/65_15/test/zsd/zsd_TCB_test.py pth/COCO_65_15.pth --out results/zsd_65_15.pkl --eval proposal
python tools/test.py configs/cattle_config/only_cattle_stone_test.py pth/cattle_weight.pth --out results/cattle.pkl --eval proposal
