import argparse
import os
import os.path as osp
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
# from mmcv.runner import get_dist_info, load_checkpoint

# from mmdet.apis import init_dist
from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.core import bbox2result
from load_plugin_lib import load_plugin_lib
load_plugin_lib()
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
from tools import engine as engine_utils
from tools import common
import pycuda.driver as cuda
import numpy as np
from functools import reduce
import warnings
import time

class TrtModel(object):
    def __init__(self, engine_path, dynamic_shape={}, dynamic_shape_value={}) -> None:
        trt.init_libnvinfer_plugins(TRT_LOGGER, '')
        self.rt = trt.Runtime(TRT_LOGGER)
        self.engine = None
        cuda.init()
        self.cfx = cuda.Device(0).make_context()
        if not os.path.exists(engine_path):
            raise Exception('tensorRT engine file not exist')
        self.engine = engine_utils.load_engine(self.rt, engine_path)
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = \
            common.allocate_buffers(self.engine, self.ctx, dynamic_shape, dynamic_shape_value)

    def infer(self, dict_input):
        for i, binding in enumerate(self.engine):
            if self.engine.binding_is_input(binding):
                input_arr = dict_input[binding]
                self.inputs[i].host = input_arr
                actual_shape = list(input_arr.shape)
                self.ctx.set_binding_shape(i, actual_shape)
        common.do_inference(self.ctx, self.bindings, self.inputs, self.outputs, self.stream)
        res = []
        for i, data in self.outputs.items():
            actual_shape = [dim for dim in self.ctx.get_binding_shape(i)]
            dsize = reduce(lambda x, y: x*y, actual_shape)
            out = data.host[:dsize].reshape(actual_shape)
            res.append(out)
        return res

    def __del__(self):
        self.cfx.pop()

def trt_test(trt_model, data_loader):
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    test_inputs = {}
    device_id = 0
    dataset = data_loader.dataset
    # warmup = {}
    # for data in data_loader:
    #     img, img_meta = data['img'][0], data['img_meta'][0].data[0][0]
    #     warmup['img'] = img.numpy()
    #     warmup['img_shape'] = np.array(img_meta['img_shape'], dtype=np.int32)
    #     warmup['scale_factor'] = np.array(img_meta['scale_factor'], dtype = np.float32)
    #     break
    # for i in range(10):
    #     det_bboxes, det_labels = trt_model.infer(warmup)
    #     result = bbox2result(det_bboxes, det_labels, 66)
    # prog_bar = mmcv.ProgressBar(len(dataset))
    results = []
    for idx, data in enumerate(data_loader):
        if idx == 5:
            prog_bar = mmcv.ProgressBar(len(dataset)-5)
        img, img_meta = data['img'][0], data['img_meta'][0].data[0][0]
        test_inputs['img'] = img.numpy()
        test_inputs['img_shape'] = np.array(img_meta['img_shape'], dtype=np.int32)
        test_inputs['scale_factor'] = np.array(img_meta['scale_factor'], dtype = np.float32)
        det_bboxes, det_labels = trt_model.infer(test_inputs)
        result = bbox2result(det_bboxes, det_labels, 66)
        # print(det_labels)
        batch_size = img.size(0)
        if idx >= 5:
            for _ in range(batch_size):
                prog_bar.update()
        results.append(result)
    return results

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='model config')
    parser.add_argument('trt_path', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    cfg = mmcv.Config.fromfile(args.config)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model and load checkpoint
    dynamic_input = {
        "img": [1,3,768,1344]
    }
    # dynamic_input_value = {
    #     "img_shape": [800, 1344,3],
    # }
    trt_model = TrtModel(args.trt_path, dynamic_shape=dynamic_input)
    outputs = trt_test(trt_model, data_loader)

    # print(len(outputs))
    # print('\nwriting results to {}'.format(args.out))
    # mmcv.dump(outputs, args.out)
    if not isinstance(outputs[0], dict):
        result_files = results2json(dataset, outputs, args.out)
        coco_eval(result_files, ['proposal'], dataset.coco)


if __name__ == '__main__':
    main()
