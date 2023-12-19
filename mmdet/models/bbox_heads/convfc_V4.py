import torch.nn as nn
import torch

from mmdet.models.bbox_heads.bbox_head_semantic_ds import BBoxSemanticHeadDynamicSeen


from ..utils import ConvModule
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


from mmdet.core import (delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy

from ..registry import HEADS

@HEADS.register_module
class ConvFCSemanticBBoxHeadV4(BBoxSemanticHeadDynamicSeen):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

                                /-> cls convs -> cls fcs -> cls
    shared convs -> shared fcs
                                \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 num_shared_convs=0,
                 num_shared_fcs=0,
                 num_semantic_convs=0,
                 num_semantic_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 semantic_dims=300,
                 semantic_num=2,
                 conv_cfg=None,
                 norm_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCSemanticBBoxHeadV4, self).__init__(*args, **kwargs)
        assert (num_shared_convs + num_shared_fcs + num_semantic_convs +
                num_semantic_fcs + num_reg_convs + num_reg_fcs > 0)
        if num_semantic_convs > 0 or num_reg_convs > 0:
            assert num_shared_fcs == 0
        if not self.with_semantic:
            assert num_semantic_convs == 0 and num_semantic_fcs == 0
        if not self.with_reg:
            assert num_reg_convs == 0 and num_reg_fcs == 0
        assert semantic_num >= 1
        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_semantic_convs = num_semantic_convs
        self.num_semantic_fcs = num_semantic_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        # add shared convs and fcs
        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        # add semantic specific branch
        self.semantic_convs, self.semantic_fcs, self.semantic_last_dim = \
            self._add_conv_fc_branch(
                self.num_semantic_convs, self.num_semantic_fcs, self.shared_out_channels)

        # add reg specific branch
        self.reg_convs, self.reg_fcs, self.reg_last_dim = \
            self._add_conv_fc_branch(
                self.num_reg_convs, self.num_reg_fcs, self.shared_out_channels)

        if self.num_shared_fcs == 0 and not self.with_avg_pool:
            if self.num_semantic_fcs == 0:
                self.semantic_last_dim *= self.roi_feat_area
            if self.num_reg_fcs == 0:
                self.reg_last_dim *= self.roi_feat_area

        self.relu = nn.ReLU(inplace=True)

        # illusion layer to mix multi-dynamic vecs
        self.illusion = nn.Linear(semantic_dims, semantic_dims,bias=False).cuda()
        self.illusion.weight.requires_grad = True
        nn.init.constant_(self.illusion.weight,0.0)
        for i in range(semantic_dims):
            nn.init.constant_(self.illusion.weight[i,i], 1.0)
        #reconstruct fc_semantic and fc_reg since input channels are changed
        if self.with_semantic:
            self.fc_semantic = nn.Linear(self.semantic_last_dim, semantic_dims)
            if self.with_decoder:
                self.d_fc_semantic = nn.Linear(semantic_dims, self.semantic_last_dim)
            if self.voc is not None:
                self.kernel_semantic = nn.Linear(self.voc.shape[1], self.vec.shape[0])  # n*300
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Linear(self.vec.shape[0], self.voc.shape[1])  # n*300
            else:
                self.kernel_semantic = nn.Linear(self.vec.shape[1], self.vec.shape[1])
                if self.with_decoder:
                    self.d_kernel_semantic = nn.Linear(self.vec.shape[1], self.vec.shape[1])  # n*300

        if self.with_reg and self.reg_with_semantic:
            self.fc_reg_sem = nn.Linear(self.reg_last_dim, semantic_dims)
            if not self.share_semantic:
                self.kernel_semantic_reg = nn.Linear(self.voc.shape[1], self.vec.shape[0])
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.num_classes, out_dim_reg)

        if self.with_reg and not self.reg_with_semantic:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = nn.Linear(self.reg_last_dim, out_dim_reg)

        self.fc_res = nn.Linear(self.vec.shape[0], self.vec.shape[0])
        # self.fc_res = nn.Linear(self.semantic_last_dim, self.vec.shape[0])

    def _add_conv_fc_branch(self,
                            num_branch_convs,
                            num_branch_fcs,
                            in_channels,
                            is_shared=False):
        """Add shared or separable branch

        convs -> avg pool (optional) -> fcs
        """
        last_layer_dim = in_channels
        # add branch specific conv layers
        branch_convs = nn.ModuleList()
        if num_branch_convs > 0:
            for i in range(num_branch_convs):
                conv_in_channels = (
                    last_layer_dim if i == 0 else self.conv_out_channels)
                branch_convs.append(
                    ConvModule(
                        conv_in_channels,
                        self.conv_out_channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg))
            last_layer_dim = self.conv_out_channels
        # add branch specific fc layers
        branch_fcs = nn.ModuleList()
        if num_branch_fcs > 0:
            # for shared branch, only consider self.with_avg_pool
            # for separated branches, also consider self.num_shared_fcs
            if (is_shared
                    or self.num_shared_fcs == 0) and not self.with_avg_pool:
                last_layer_dim *= self.roi_feat_area
            for i in range(num_branch_fcs):
                fc_in_channels = (
                    last_layer_dim if i == 0 else self.fc_out_channels)
                branch_fcs.append(
                    nn.Linear(fc_in_channels, self.fc_out_channels))
            last_layer_dim = self.fc_out_channels
        return branch_convs, branch_fcs, last_layer_dim

    def init_weights(self):
        super(ConvFCSemanticBBoxHeadV4, self).init_weights()
        for module_list in [self.shared_fcs, self.semantic_fcs, self.reg_fcs]:
            for m in module_list.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, res_feats=None, context_feats=None, return_feats=False, resturn_center_feats=False, bg_vector=None):
        # shared part
        if self.num_shared_convs > 0:
            for conv in self.shared_convs:
                x = conv(x)

        if self.num_shared_fcs > 0:
            if self.with_avg_pool:
                x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            for fc in self.shared_fcs:
                x = self.relu(fc(x))
        # separate branches
        x_semantic = x
        x_reg = x

        for conv in self.semantic_convs:
            x_semantic = conv(x_semantic)
        if x_semantic.dim() > 2:
            if self.with_avg_pool:
                x_semantic = self.avg_pool(x_semantic)
            x_semantic = x_semantic.view(x_semantic.size(0), -1)
        for fc in self.semantic_fcs:
            x_semantic = self.relu(fc(x_semantic))

        for conv in self.reg_convs:
            x_reg = conv(x_reg)
        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.view(x_reg.size(0), -1)
        for fc in self.reg_fcs:
            x_reg = self.relu(fc(x_reg))

        if self.with_semantic:
            semantic_feature = self.fc_semantic(x_semantic)
            # fusion new word vec            
            self.vec_dy = self.illusion(self.vec_fix.t()).t()
            if self.sync_bg:
                with torch.no_grad():
                    self.vec_fix[:,0] = bg_vector
                    self.vec_dy[:,0] = bg_vector
                    if not self.seen_class:
                        self.vec_unseen[:, 0] = bg_vector
            
            
            #semantic_vec = self.fusion_layer(vecs_all.t()).t() # 不用经过线性层  和向量做矩阵乘法 之举
            if self.voc is not None:

                semantic_score = torch.mm(semantic_feature, self.voc)
                if self.semantic_norm:
                    semantic_score_norm = torch.norm(semantic_score, p=2, dim=1).unsqueeze(1).expand_as(semantic_score)
                    semantic_score = semantic_score.div(semantic_score_norm + 1e-5)
                    temp_norm = torch.norm(self.kernel_semantic.weight.data, p=2, dim=1).unsqueeze(1).expand_as(self.kernel_semantic.weight.data)
                    self.kernel_semantic.weight.data = self.kernel_semantic.weight.data.div(temp_norm + 1e-5)
                    semantic_score = self.kernel_semantic(semantic_score) * 20.0
                else:
                    semantic_score = self.kernel_semantic(semantic_score)
                if self.with_decoder:
                    d_semantic_score = self.d_kernel_semantic(semantic_score)
                    d_semantic_feature = torch.mm(d_semantic_score, self.voc.t())
                    d_semantic_feature = self.d_fc_semantic(d_semantic_feature)

                semantic_score_fix = torch.mm(semantic_score, self.vec_fix)
                semantic_score_dy = torch.mm(semantic_score, self.vec_dy)
                semantic_score = torch.max(semantic_score_dy,semantic_score_fix)
            else:
                semantic_score = self.kernel_semantic(self.vec_fix)
                semantic_score = torch.tanh(semantic_score)
                semantic_score = torch.mm(semantic_feature, semantic_score)
        else:
            semantic_score = None
        if self.with_reg and not self.reg_with_semantic:
            bbox_pred = self.fc_reg(x_reg)
        elif self.with_reg and self.reg_with_semantic:
            semantic_reg_feature = self.fc_reg_sem(x_reg)
            if not self.share_semantic:
                semantic_reg_score = torch.mm(self.kernel_semantic_reg(self.voc), self.vec_fix)
            else:
                semantic_reg_score = torch.mm(self.kernel_semantic(self.voc), self.vec_fix)
            semantic_reg_score = torch.tanh(semantic_reg_score)
            semantic_reg_score = torch.mm(semantic_reg_feature, semantic_reg_score)
            bbox_pred = self.fc_reg(semantic_reg_score)
        else:
            bbox_pred = None
        if self.with_decoder:
            return semantic_score, bbox_pred, x_semantic, d_semantic_feature
        else:
            return semantic_score, bbox_pred

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
            loss_encoder_decoder = self.loss_ed(x_semantic, d_feature)
            losses['bbox_loss_ed'] = loss_encoder_decoder
        #TODO npair loss
        loss_npair = self.loss_npair(self.vec_dy)  #self.vec[0] bg vec
        losses['loss_npair'] = loss_npair

        return losses
#TODO
    @force_fp32(apply_to=('semantic_score', 'bbox_pred'))
    def get_det_bboxes(self,
                       rois,
                       semantic_score,
                       bbox_pred,
                       img_shape,
                       scale_factor,
                       rescale=False,
                       cfg=None):
        if isinstance(semantic_score, list):
            semantic_score = sum(semantic_score) / float(len(semantic_score))
        scores = F.softmax(semantic_score, dim=1) if semantic_score is not None else None
        semantic_vec = self.vec_fix
        semantic_unseen = self.illusion(self.vec_unseen.t()).t()
        if self.gzsd:
            seen_scores = torch.mm(scores, semantic_vec.t())
            seen_scores = torch.mm(seen_scores, semantic_vec)
            seen_bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                     self.target_stds, img_shape)

            unseen_scores = torch.mm(scores, semantic_vec.t())
            unseen_scores_fix = torch.mm(unseen_scores, self.vec_unseen)
            unseen_scores_dy  = torch.mm(unseen_scores, semantic_unseen)
            unseen_scores  =torch.max(unseen_scores_dy, unseen_scores_fix)
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
            scores = torch.mm(scores, semantic_vec.t())
            scores = torch.mm(scores, semantic_vec)
        # TODO ZSD  open these lines when unseen inference
        if not self.seen_class:
            scores = torch.mm(scores, semantic_vec.t())
            unseen_scores_fix = torch.mm(scores, self.vec_unseen)
            unseen_scores_dy  = torch.mm(scores, semantic_unseen)
            scores  =torch.max(unseen_scores_dy, unseen_scores_fix)
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
@HEADS.register_module
class SharedFCSemanticBBoxHeadDSV4(ConvFCSemanticBBoxHeadV4):

    def __init__(self, num_fcs=2, fc_out_channels=1024, *args, **kwargs):
        assert num_fcs >= 1
        super(SharedFCSemanticBBoxHeadDSV4, self).__init__(
            num_shared_convs=0,
            num_shared_fcs=num_fcs,
            num_semantic_convs=0,
            num_semantic_fcs=0,
            num_reg_convs=0,
            num_reg_fcs=0,
            fc_out_channels=fc_out_channels,
            *args,
            **kwargs)
