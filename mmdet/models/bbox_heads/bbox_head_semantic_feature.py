import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

import numpy as np

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from ..losses import LSoftmaxLinear

@HEADS.register_module
class BBoxSemanticHeadFeature(nn.Module):
    """Simplest RoI head, with only two fc layers for semantic and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_reg=True,
                 with_semantic=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_classes=66,
                 semantic_dims=300,
                 seen_class=True,
                 gzsd=False,
                 reg_with_semantic=False,
                 share_semantic=False,
                 voc_path=None,
                 vec_path=None,
                 use_lsoftmax=False,
                 with_decoder=False,
                 sync_bg=False,
                 semantic_norm=False,
                 use_topk = False,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_semantic=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_ed=dict(type='MSELoss', loss_weight=0.5),
                 loss_npair=dict(type='NPairLoss',loss_weight=0.2)
                 ):
        super(BBoxSemanticHeadFeature, self).__init__()
        assert with_reg or with_semantic
        self.seen_class = seen_class
        self.gzsd = gzsd
        self.reg_with_semantic = reg_with_semantic
        self.share_semantic = share_semantic
        self.with_avg_pool = with_avg_pool
        self.with_reg = with_reg
        self.with_semantic = with_semantic
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False
        self.use_lsoftmax = use_lsoftmax
        self.with_decoder = with_decoder
        self.semantic_norm = semantic_norm
        self.topk = use_topk
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_semantic = build_loss(loss_semantic)
        self.loss_ed = build_loss(loss_ed)
        self.loss_npair = build_loss(loss_npair)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)
        if self.with_semantic:
            self.fc_semantic = nn.Linear(self.in_channels, semantic_dims)
            # voc = np.loadtxt('MSCOCO/vocabulary_w2v.txt', dtype='float32', delimiter=',')
            if voc_path is not None:
                voc = np.loadtxt(voc_path, dtype='float32', delimiter=',')
            else:
                voc = None
            self.projection_layer = nn.Linear(semantic_dims, semantic_dims,False)
            nn.init.eye_(self.projection_layer.weight)
            # vec = np.loadtxt('MSCOCO/word_w2v.txt', dtype='float32', delimiter=',')
            vec_load = np.loadtxt(vec_path, dtype='float32', delimiter=',')
            # if self.seen_class:
            vec = vec_load[:, :num_classes]
            # else:
            vec_unseen = np.concatenate([vec_load[:, 0:1], vec_load[:, num_classes:]], axis=1)
            vec = torch.tensor(vec, dtype=torch.float32)
            if voc is not None:
                voc = torch.tensor(voc, dtype=torch.float32)
            vec_unseen = torch.tensor(vec_unseen, dtype=torch.float32)
            self.vec = vec.cuda()  # 300*n
            if voc is not None:
                self.voc = voc.cuda()  # 300*66
            else:
               self.voc = None
            self.vec_unseen = vec_unseen.cuda()

            if self.voc is not None:
                self.kernel_semantic = nn.Linear(self.voc.shape[1], self.vec.shape[0]) #n*300

        if self.use_lsoftmax:
            self.lsoftmax = LSoftmaxLinear(num_classes, num_classes, margin=4)

        self.sync_bg = sync_bg
        self.debug_imgs = None

    def init_weights(self):
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)
        if self.with_semantic:
            nn.init.normal_(self.fc_semantic.weight, 0, 0.001)
            nn.init.constant_(self.fc_semantic.bias, 0)
            if self.voc is not None:
                nn.init.normal_(self.kernel_semantic.weight, 0, 0.001)
                nn.init.constant_(self.kernel_semantic.bias, 0)

    @auto_fp16()
    def forward(self, x):
        if self.with_avg_pool:
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        bbox_pred = self.fc_reg(x) if self.with_reg else None
        if self.with_semantic:
            semantic_feature = self.fc_semantic(x)
            semantic_score = torch.mm(self.kernel_semantic(self.voc), self.vec)
            semantic_score = torch.tanh(semantic_score)
            semantic_score = torch.mm(semantic_feature, semantic_score)
        else:
            semantic_score = None
        return semantic_score, bbox_pred

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        semantic_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return semantic_reg_targets

    @force_fp32(apply_to=('semantic_score', 'bbox_pred'))
    def loss(self,
             semantic_score,
             bbox_pred,
             labels,
             label_weights,
             bbox_targets,
             bbox_weights,
             x_semantic=None,
             d_feature=None,
             reduction_override=None):
        losses = dict()
        if semantic_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_semantic'] = self.loss_semantic(
                semantic_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(semantic_score, labels)
        if bbox_pred is not None:
            pos_inds = labels > 0
            if self.reg_class_agnostic:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), 4)[pos_inds]
            else:
                pos_bbox_pred = bbox_pred.view(bbox_pred.size(0), -1,
                                               4)[pos_inds, labels[pos_inds]]
            losses['loss_bbox'] = self.loss_bbox(
                pos_bbox_pred,
                bbox_targets[pos_inds],
                bbox_weights[pos_inds],
                avg_factor=bbox_targets.size(0),
                reduction_override=reduction_override)
        if self.with_decoder and x_semantic is not None and d_feature is not None:
            if isinstance(d_feature, (list)):
                loss_encoder_decoder = 0
                for decoder_res in d_feature: 
                    loss_encoder_decoder += self.loss_ed(x_semantic, decoder_res)
            else:
                loss_encoder_decoder = self.loss_ed(x_semantic, d_feature)
            losses['bbox_loss_ed'] = loss_encoder_decoder
        if self.loss_npair is not None:
            loss_npair = self.loss_npair(self.projection_layer(self.vec[:,1:].t()).t())
            losses['loss_npair'] = loss_npair
        return losses
#TODO

    def topkvalue(self,res1,res2,n):
        ratio = 1.0/n+1
        weight = [2.0 - ratio * i for i in range(n)]
        _,index1 = res1.topk(n)
        for i in range(n):
            res1[index1[i]] = res1[index1[i]] * weight[i]
        
        _,index2 = res2.topk(n)
        for i in range(n):
            res2[index2[i]] = res2[index2[i]] * weight[i]
        res = res1 + res2
        res = F.softmax(res, dim=-1)
        return res

    def matrix_max(self,m1,m2):
        res = torch.zeros_like(m1)
        n,d = m1.size()
        for i in range(n):
            res[i,:] = torch.max(m1[i,:],m2[i,:])
        return res
    @force_fp32(apply_to=('semantic_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       projected_feature,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        
        # scores = LSoftmaxLinear(semantic_score, dim=1) if semantic_score is not None else None

        if self.gzsd:
            #seen_scores = torch.mm(scores, self.vec.t()) # 得到feature
            
            seen_bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
            if self.topk:
                seen_scores_fix = torch.mm(projected_feature, self.vec)
                seen_scores_dy = torch.mm(projected_feature, 
                                                self.projection_layer(self.vec.t()).t())
                seen_scores = self.topkvalue(seen_scores_fix, seen_scores_dy, 3)
            else:
                seen_scores_fix = torch.mm(projected_feature, self.vec)
                seen_scores_dy = torch.mm(projected_feature, self.projection_layer(self.vec.t()).t())
                seen_scores =  F.softmax(self.matrix_max(seen_scores_fix , seen_scores_dy), dim=1)
            #unseen_scores = torch.mm(scores, self.vec.t())
            
            if self.topk:
                unseen_scores_fix = torch.mm(projected_feature, self.vec_unseen)
                unseen_scores_dy = torch.mm(projected_feature, 
                                                self.projection_layer(self.vec_unseen.t()).t())

                unseen_scores = self.topkvalue(unseen_scores_fix, unseen_scores_dy,3)
            else:
                unseen_scores_fix =  torch.mm(projected_feature, self.vec_unseen)
                unseen_scores_dy = torch.mm(projected_feature, self.projection_layer(self.vec_unseen.t()).t())                                       
                unseen_scores =  F.softmax(self.matrix_max(unseen_scores_fix ,unseen_scores_dy), dim=1)
            unseen_bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                     self.target_stds, img_shape)

            if rescale:
                if isinstance(scale_factor, float):
                    seen_bboxes /= scale_factor
                    unseen_bboxes /= scale_factor
                else:
                    seen_bboxes /= torch.from_numpy(scale_factor).to(seen_bboxes.device)
                    unseen_bboxes /= torch.from_numpy(scale_factor).to(unseen_bboxes.device)

            if cfg is None:
                return [seen_bboxes, unseen_bboxes], [seen_scores, unseen_scores]
            else:
                seen_det_bboxes, seen_det_labels = multiclass_nms(seen_bboxes, seen_scores,
                                                        # 0.2, cfg.nms,
                                                        0.05, cfg.nms,
                                                        cfg.max_per_img)
                unseen_det_bboxes, unseen_det_labels = multiclass_nms(unseen_bboxes, unseen_scores,
                                                                  0.05, cfg.nms,
                                                                  cfg.max_per_img)
                # unseen_det_labels += 65
                # unseen_det_labels += 48
                unseen_det_labels += (self.num_classes - 1)

                det_bboxes = torch.cat([seen_det_bboxes, unseen_det_bboxes], dim=0)
                det_labels = torch.cat([seen_det_labels, unseen_det_labels], dim=0)
                # return [seen_det_bboxes, unseen_det_bboxes], [seen_det_labels, unseen_det_labels]
                return det_bboxes, det_labels

        if self.seen_class:
            #scores = torch.mm(scores, self.vec.t())
            if self.topk:
                seen_scores_fix = torch.mm(projected_feature, self.vec)
                seen_scores_dy = torch.mm(projected_feature, 
                                                self.projection_layer(self.vec.t()).t())
                                                                               
                seen_scores = self.topkvalue(seen_scores_fix, seen_scores_dy, 3)
            else:
                seen_scores_fix = torch.mm(projected_feature, self.vec)
                seen_scores_dy = torch.mm(projected_feature, self.projection_layer(self.vec.t()).t())
                seen_scores =  self.matrix_max(seen_scores_fix , seen_scores_dy)
            scores = seen_scores
        # TODO ZSD  open these lines when unseen inference
        if not self.seen_class:
            #scores = torch.mm(scores, self.vec.t())
            if self.topk:
                unseen_scores_fix = torch.mm(projected_feature, self.vec_unseen)
                unseen_scores_dy = torch.mm(projected_feature, 
                                                self.projection_layer(self.vec_unseen.t()).t())
                unseen_scores = self.topkvalue(unseen_scores_fix, unseen_scores_dy,3)
            else:
                unseen_scores_fix =  torch.mm(projected_feature, self.vec_unseen)
                unseen_scores_dy = torch.mm(projected_feature, self.projection_layer(self.vec_unseen.t()).t())                                       
                unseen_scores = self.matrix_max(unseen_scores_fix ,unseen_scores_dy)
            scores =  unseen_scores
            # toster_score = torch.argmax(scores[:, 1:], dim=1) == 13
            # for i, vailed in enumerate(toster_score):
            #     if vailed:
            #         scores[i, :] = 0.0
            # dog_score = torch.argmax(scores[:, 1:], dim=1) == 1
            # for i, vailed in enumerate(dog_score):
            #     if vailed and torch.max(scores[i, :]) <= 0.15:
            #         scores[i, :] = 0.0


            # topK scores
            # T = 5
            # mask = torch.ones_like(scores)
            # mask[:, T:] = 0.0
            # # print(scores.size())
            # sorted_score, _ = torch.sort(scores, dim=1, descending=True)
            # sorted_score_arg = torch.argsort(scores, dim=1, descending=True)
            # sorted_score = sorted_score.mul(mask)
            #
            # restroed_score = mask
            # for i in range(scores.shape[0]):
            #     restroed_score[i, sorted_score_arg[i, :]] = sorted_score[i, :]
            #
            # unseen_pd = torch.mm(restroed_score, self.vec.t())
            # scores = torch.mm(unseen_pd, self.vec_unseen)

            # toster_score = torch.argmax(scores[:, 1:], dim=1) == 13
            # for i, vailed in enumerate(toster_score):
            #     if vailed:
            #         scores[i, :] = 0.0


        if bbox_pred is not None:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_shape)
        else:
            bboxes = rois[:, 1:].clone()
            if img_shape is not None:
                bboxes[:, [0, 2]].clamp_(min=0, max=img_shape[1] - 1)
                bboxes[:, [1, 3]].clamp_(min=0, max=img_shape[0] - 1)

        if rescale:
            if isinstance(scale_factor, float):
                bboxes /= scale_factor
            else:
                bboxes /= torch.from_numpy(scale_factor).to(bboxes.device)

        if cfg is None:
            return bboxes, scores
        else:
            det_bboxes, det_labels = multiclass_nms(bboxes, scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)

            return det_bboxes, det_labels
    

    @force_fp32(apply_to=('bbox_preds', ))
    def refine_bboxes(self, rois, labels, bbox_preds, pos_is_gts, img_metas):
        """Refine bboxes during training.

        Args:
            rois (Tensor): Shape (n*bs, 5), where n is image number per GPU,
                and bs is the sampled RoIs per image.
            labels (Tensor): Shape (n*bs, ).
            bbox_preds (Tensor): Shape (n*bs, 4) or (n*bs, 4*#class).
            pos_is_gts (list[Tensor]): Flags indicating if each positive bbox
                is a gt bbox.
            img_metas (list[dict]): Meta info of each image.

        Returns:
            list[Tensor]: Refined bboxes of each image in a mini-batch.
        """
        img_ids = rois[:, 0].long().unique(sorted=True)
        assert img_ids.numel() == len(img_metas)

        bboxes_list = []
        for i in range(len(img_metas)):
            inds = torch.nonzero(rois[:, 0] == i).squeeze()
            num_rois = inds.numel()

            bboxes_ = rois[inds, 1:]
            label_ = labels[inds]
            bbox_pred_ = bbox_preds[inds]
            img_meta_ = img_metas[i]
            pos_is_gts_ = pos_is_gts[i]

            bboxes = self.regress_by_class(bboxes_, label_, bbox_pred_,
                                           img_meta_)
            # filter gt bboxes
            pos_keep = 1 - pos_is_gts_
            keep_inds = pos_is_gts_.new_ones(num_rois)
            keep_inds[:len(pos_is_gts_)] = pos_keep

            bboxes_list.append(bboxes[keep_inds])

        return bboxes_list

    @force_fp32(apply_to=('bbox_pred', ))
    def regress_by_class(self, rois, label, bbox_pred, img_meta):
        """Regress the bbox for the predicted class. Used in Cascade R-CNN.

        Args:
            rois (Tensor): shape (n, 4) or (n, 5)
            label (Tensor): shape (n, )
            bbox_pred (Tensor): shape (n, 4*(#class+1)) or (n, 4)
            img_meta (dict): Image meta info.

        Returns:
            Tensor: Regressed bboxes, the same shape as input rois.
        """
        assert rois.size(1) == 4 or rois.size(1) == 5

        if not self.reg_class_agnostic:
            label = label * 4
            inds = torch.stack((label, label + 1, label + 2, label + 3), 1)
            bbox_pred = torch.gather(bbox_pred, 1, inds)
        assert bbox_pred.size(1) == 4

        if rois.size(1) == 4:
            new_rois = delta2bbox(rois, bbox_pred, self.target_means,
                                  self.target_stds, img_meta['img_shape'])
        else:
            bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                self.target_stds, img_meta['img_shape'])
            new_rois = torch.cat((rois[:, [0]], bboxes), dim=1)

        return new_rois
