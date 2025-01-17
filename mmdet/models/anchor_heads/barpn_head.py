import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from mmdet.core import delta2bbox
# from mmdet.ops import nms
from mmcv.ops import nms
from ..registry import HEADS
from .ba_anchor_head import BackgroundAwareAnchorHead


@HEADS.register_module
class BackgroundAwareRPNHead(BackgroundAwareAnchorHead):

    def __init__(self, in_channels, freeze=False, **kwargs):
        self.freeze=freeze
        super(BackgroundAwareRPNHead, self).__init__(2, in_channels, **kwargs)

    def _init_layers(self):
        self.rpn_conv = nn.Conv2d(
            self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls_conv_T = nn.Conv2d(self.feat_channels, self.semantic_dims, 1)
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 4, 1)
        if self.freeze:
            for m in [self.rpn_conv, self.rpn_cls, self.rpn_reg]:
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self):
        normal_init(self.rpn_conv, std=0.01)
        normal_init(self.rpn_cls_conv_T, std=0.01)
        normal_init(self.rpn_reg, std=0.01)
        normal_init(self.vec_fb)
        with torch.no_grad():
            self.vec_fb.weight.data[0] = self.vec_bg_weight.unsqueeze(-1).unsqueeze(-1)
            self.vec_fb.weight.data[2] = self.vec_bg_weight.unsqueeze(-1).unsqueeze(-1)
            self.vec_fb.weight.data[4] = self.vec_bg_weight.unsqueeze(-1).unsqueeze(-1)

    def forward_single(self, x):
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        # B C(900) W H
        rpn_cls_score = self.rpn_cls_conv_T(x)
        if self.voc:
            rpn_cls_score = self.voc_conv(rpn_cls_score)

        rpn_cls_score = self.vec_fb(rpn_cls_score)

        rpn_bbox_pred = self.rpn_reg(x)
        if self.sync_bg:
            return rpn_cls_score, rpn_bbox_pred, \
                   (self.vec_fb.weight.data[0] + self.vec_fb.weight.data[2]+ self.vec_fb.weight.data[4]) / 3.0
        return rpn_cls_score, rpn_bbox_pred

    def loss(self,
             cls_scores,
             bbox_preds,
             gt_bboxes,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        losses = super(BackgroundAwareRPNHead, self).loss(
            cls_scores,
            bbox_preds,
            gt_bboxes,
            None,
            img_metas,
            cfg,
            gt_bboxes_ignore=gt_bboxes_ignore)
        return dict(
            loss_rpn_cls=losses['loss_cls'], loss_rpn_bbox=losses['loss_bbox'])

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        mlvl_proposals = []
        for idx in range(len(cls_scores)):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]
            anchors = mlvl_anchors[idx]
            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 1]
            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if cfg.nms_pre > 0 and scores.shape[0] > cfg.nms_pre:
                _, topk_inds = scores.topk(cfg.nms_pre)
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]
                scores = scores[topk_inds]
            proposals = delta2bbox(anchors, rpn_bbox_pred, self.target_means,
                                   self.target_stds, img_shape)
            if cfg.min_bbox_size > 0:
                w = proposals[:, 2] - proposals[:, 0] + 1
                h = proposals[:, 3] - proposals[:, 1] + 1
                valid_inds = torch.nonzero((w >= cfg.min_bbox_size) &
                                           (h >= cfg.min_bbox_size)).squeeze()
                proposals = proposals[valid_inds, :]
                scores = scores[valid_inds]
            _, nms_indices = nms(proposals, scores, cfg.nms_thr)
            proposals = torch.cat([proposals, scores.unsqueeze(-1)], dim=-1)
            # proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[nms_indices, :]
            proposals = proposals[:cfg.nms_post, :]
            mlvl_proposals.append(proposals)
        proposals = torch.cat(mlvl_proposals, 0)
        if cfg.nms_across_levels:
            proposals, _ = nms(proposals, cfg.nms_thr)
            proposals = proposals[:cfg.max_num, :]
        else:
            scores = proposals[:, 4]
            num = min(cfg.max_num, proposals.shape[0])
            _, topk_inds = scores.topk(num)
            proposals = proposals[topk_inds, :]
        return proposals
