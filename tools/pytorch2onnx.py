import argparse
from collections import OrderedDict

import mmcv
import torch
from mmdet.models import build_detector
from mmdet.datasets import build_dataloader, build_dataset
from mmcv.runner import load_checkpoint
from mmdet.core import bbox2roi, bbox2result
from mmdet.core import coco_eval, results2json
import torch.nn as nn
from mmcv.parallel import MMDataParallel
import onnxruntime
import warnings
from torch.onnx import OperatorExportTypes
from onnx.shape_inference import infer_shapes
import onnx
from torch.onnx import register_custom_op_symbolic
from mmdet.ops.roi_align.roi_align import roi_align_symb

class OnnxModel(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model
        
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x= self.model.backbone(img)
        x = self.model.neck(x)
        return x
    
    def get_bboxes(self, cls_scores, bbox_preds, img_shape, scale_factor, cfg,
                   rescale=False):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        mlvl_anchors = [
            self.model.rpn_head.anchor_generators[i].grid_anchors(
                cls_scores[i].size()[-2:],
                self.model.rpn_head.anchor_strides[i],
                device="cuda") for i in range(num_levels)
        ]
        result_list = []
        cls_score_list = [
            cls_scores[i][0].detach() for i in range(num_levels)
        ]
        bbox_pred_list = [
            bbox_preds[i][0].detach() for i in range(num_levels)
        ]
        proposals = self.model.rpn_head.get_bboxes_single(cls_score_list, bbox_pred_list,
                                           mlvl_anchors, img_shape,
                                           scale_factor, cfg, rescale)
        result_list.append(proposals)
        return result_list
    
    def simple_test_rpn(self,
                        x,
                        img_shape,
                        scale_factor,
                        rpn_test_cfg):
        rpn_outs = self.model.rpn_head(x)
        bg_vector = rpn_outs[-1]
        rpn_outs = rpn_outs[:-1]
        proposal_list = self.get_bboxes(rpn_outs[0], rpn_outs[1], img_shape=img_shape, scale_factor=scale_factor, cfg=rpn_test_cfg)
        return proposal_list, bg_vector
    
    def simple_test_bboxes(self,
                           x,
                           img_shape,
                           scale_factor,
                           proposals,
                           rcnn_test_cfg,
                           bg_vector=None,
                           with_decoder=False,
                           rescale=False):
        """Test only det bboxes without augmentation."""
        rois = bbox2roi(proposals)
        roi_feats = self.model.bbox_roi_extractor(
            x[:len(self.model.bbox_roi_extractor.featmap_strides)], rois)
        if self.model.with_shared_head:
            roi_feats = self.model.shared_head(roi_feats)
        if with_decoder:
            cls_score, bbox_pred, _, _ = self.model.bbox_head(roi_feats, bg_vector=bg_vector)
        else:
            cls_score, bbox_pred = self.model.bbox_head(roi_feats, bg_vector=bg_vector)
        det_bboxes, det_labels = self.model.bbox_head.get_det_bboxes(
            rois,
            cls_score,
            bbox_pred,
            img_shape,
            scale_factor,
            rescale=rescale,
            cfg=rcnn_test_cfg)
        return det_bboxes, det_labels
        
    def forward(self,
                img,
                img_shape,
                scale_factor,
                ):
        x = self.extract_feat(img)
        proposal_list, bg_vector= self.simple_test_rpn(
            x, img_shape, scale_factor, self.model.test_cfg.rpn)
        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_shape, scale_factor, proposal_list, self.model.test_cfg.rcnn, with_decoder=True, rescale=True, bg_vector=bg_vector[0].view(-1,))
        # bbox_results = bbox2result(det_bboxes, det_labels,
        #     self.model.bbox_head.num_classes)
        return det_bboxes, det_labels
    
def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        img, img_meta = data['img'][0], data['img_meta'][0].data[0][0]
        with torch.no_grad():
            result = model(img, img_meta['img_shape'], img_meta['scale_factor'])
        results.append(result)

        if show:
            # model.module.show_result(data, result)
            model.module.show_result(data, result, score_thr=0.30)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results

def convert(src, dst, config):
    """Convert pytorch model to onnx model."""
    cfg = mmcv.Config.fromfile(config)
    cfg.model.pretrained = None
    device_id = 0
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    checkpoint = load_checkpoint(model, src, map_location='cuda:0')

    onnx_model = OnnxModel(model)
    onnx_model = onnx_model.to(device_id)
    onnx_model.eval()
    # onnx_model = MMDataParallel(onnx_model, device_ids=[0])
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    
    dummy_inputs = {}
    with torch.no_grad():
        # ress=[]
        # prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            img, img_meta = data['img'][0], data['img_meta'][0].data[0][0]
            dummy_inputs['img'] = img.type(torch.float32).to(device_id)
            dummy_inputs['img_shape'] = torch.tensor(img_meta['img_shape'], dtype=torch.int32).to(device_id)
            dummy_inputs['scale_factor'] = torch.tensor(img_meta['scale_factor'], dtype=torch.float32).to(device_id)
            det_bboxes, det_labels = onnx_model(dummy_inputs['img'], dummy_inputs['img_shape'], dummy_inputs['scale_factor'])
            break
            # res = bbox2result(det_bboxes, det_labels, 66)
            # ress.append(res)
            # prog_bar.update()
        # result_files = results2json(dataset, ress, 'test_out')
        # coco_eval(result_files, ['proposal'], dataset.coco)

        output_names = ["bbox", "labels"]
        dynamic_axes = {
            "img": {2: "width", 3: "height"},
        }
        # register_custom_op_symbolic("mmdet::roi_align", roi_align_symb, 15)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            with open(dst, 'wb') as f:
                print(f"Exporting onnx model to {dst}...")
                torch.onnx.export(
                    onnx_model,
                    tuple(dummy_inputs.values()),
                    f,
                    export_params=True,
                    verbose=True,
                    operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
                    opset_version=15,
                    do_constant_folding=False,
                    input_names=list(dummy_inputs.keys()),
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    custom_opsets={"mmdet": 15}
                )
        onnx_model = onnx.load(dst)
        onnx_model = infer_shapes(onnx_model)
        onnx.save(onnx_model, dst)
        print("export done")
        results = []
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(dst, providers=providers)
        prog_bar = mmcv.ProgressBar(len(dataset))
        for data in data_loader:
            img, img_meta = data['img'][0], data['img_meta'][0].data[0][0]
            dummy_inputs['img'] = img.to(device_id)
            dummy_inputs['img_shape'] = torch.tensor(img_meta['img_shape'], dtype=torch.int32).to(device_id)
            dummy_inputs['scale_factor'] = torch.tensor(img_meta['scale_factor'], dtype=torch.float32).to(device_id)
            ort_inputs = {k: v.cpu().numpy() for k, v in dummy_inputs.items()}
            det_bboxes, det_labels = ort_session.run(None, ort_inputs)
            result = bbox2result(det_bboxes, det_labels, 66)
            results.append(result)
            batch_size = img.size(0)
            for _ in range(batch_size):
                prog_bar.update()
        result_files = results2json(dataset, results, 'test_out')
        coco_eval(result_files, ['proposal'], dataset.coco)

def main():
    parser = argparse.ArgumentParser(description='Convert model keys')
    parser.add_argument('config', help='model config')
    parser.add_argument('src', help='src pytorch model path')
    parser.add_argument('dst', help='dst onnx model path')

    args = parser.parse_args()
    convert(args.src, args.dst, args.config)


if __name__ == '__main__':
    main()
