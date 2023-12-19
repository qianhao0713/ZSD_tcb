import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import sklearn.linear_model as models
import numpy as np

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from ..utils import ConvModule
from ..VIT import ViT

@HEADS.register_module
class MESHead(nn.Module):
    """Simplest RoI head, with only two fc layers for semantic and
    regression respectively"""

    def __init__(self,
                 with_avg_pool=False,
                 with_reg=True,
                 roi_feat_size=7,
                 in_channels=256,
                 num_shared_fcs=2,
                 num_shared_convs=0,
                 num_seen_classes=66,
                 semantic_dims=300,
                 fc_out_channels=1024,
                 base_space_dimension=128,
                 num_space=3,
                 more_larger_space=True,
                 seen_class=True,
                 gzsd=False,
                 share_semantic=False,
                 voc_path=None,
                 vec_path=None,
                 with_decoder=False,
                 sync_bg=False,
                 inference_multi=False,
                 target_means=[0., 0., 0., 0.],
                 target_stds=[0.1, 0.1, 0.2, 0.2],
                 reg_class_agnostic=False,
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1.0),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 loss_con=dict(
                     type='ContrastiveLoss',
                     loss_weight=1.0,
                     tal=0.1,
                     foreground=True),
                 loss_con_sem=dict(
                     type='ContrastiveSemLoss',
                     loss_weight=1.0,
                     tal=0.1,
                     foreground=True),
                 loss_val=dict(
                     type='CrossEntropyLoss',
                     loss_weight=1.0)
                 ):
        super(MESHead, self).__init__()
        self.seen_class = seen_class
        self.gzsd = gzsd
        self.share_semantic = share_semantic
        self.with_avg_pool = with_avg_pool
        self.with_reg = with_reg
        self.roi_feat_size = _pair(roi_feat_size)
        self.roi_feat_area = self.roi_feat_size[0] * self.roi_feat_size[1]
        self.in_channels = in_channels
        self.num_seen_classes = num_seen_classes
        self.num_classes = num_seen_classes
        self.target_means = target_means
        self.target_stds = target_stds
        self.reg_class_agnostic = reg_class_agnostic
        self.fp16_enabled = False
        self.with_decoder = with_decoder
        self.num_shared_fcs = num_shared_fcs
        self.num_shared_convs = num_shared_convs
        self.semantic_dim = semantic_dims
        self.fc_out_channels = fc_out_channels
        self.base_space_dimension = base_space_dimension
        self.num_space = num_space
        self.sync_bg = sync_bg
        self.inference_multi = inference_multi
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.loss_val = build_loss(loss_val)
        self.loss_con= build_loss(loss_con)
        self.loss_con_sem = build_loss(loss_con_sem)

        self.relu = nn.ReLU(inplace=True)

        in_channels = self.in_channels
        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(self.roi_feat_size)
        else:
            in_channels *= self.roi_feat_area

        if self.with_reg:
            out_dim_reg = 4 if reg_class_agnostic else 4 * num_seen_classes
            self.fc_reg = nn.Linear(in_channels, out_dim_reg)

        self.shared_convs, self.shared_fcs, last_layer_dim = \
            self._add_conv_fc_branch(
                self.num_shared_convs, self.num_shared_fcs, self.in_channels,
                True)
        self.shared_out_channels = last_layer_dim

        if voc_path is not None:
            voc = np.loadtxt(voc_path, dtype='float32', delimiter=',')
        else:
            voc = None
        vec_load = np.loadtxt(vec_path, dtype='float32', delimiter=',')
        # if self.seen_class:
        vec = vec_load[:, :num_seen_classes]
        vec = torch.tensor(vec, dtype=torch.float32)
        self.vec = vec.cuda()
        # vec = F.normalize(vec,2,0)
        
        # else:
        vec_unseen = np.concatenate([vec_load[:, 0:1], vec_load[:, num_seen_classes:]], axis=1)
        
        if voc is not None:
            voc = torch.tensor(voc, dtype=torch.float32)
            voc = F.normalize(voc,2,0)
        vec_unseen = torch.tensor(vec_unseen, dtype=torch.float32)
        # vec_unseen = F.normalize(vec_unseen,2,0)
        
        if voc is not None:
            self.voc = voc.cuda()  # 300*66
        else:
            self.voc = None
        self.vec_unseen = vec_unseen.cuda()


        FFN = nn.ModuleList()
        for j in range(self.num_shared_fcs):
            in_channel = (self.in_channels * self.roi_feat_area if j == 0 else last_layer_dim)
            layer = nn.Linear(in_channel,last_layer_dim)
            FFN.append(layer)
        self.FFN = FFN.cuda()

        # vision2embed semantic2embed attention layer init
        self.v2e_list = nn.ModuleList()
        self.s2e_list = nn.ModuleList()
        self.voc_att_list = nn.ModuleList()
        base = 1
        for i in range(self.num_space):
            space_dim = self.base_space_dimension * base
            print("space "+str(i)+" dimension:"+" "+str(space_dim))
            v2e = nn.Sequential(
                nn.Linear(last_layer_dim, space_dim*2),
                nn.Linear(space_dim*2, space_dim),
            )
            # s2e = nn.Sequential(
            #     nn.Linear(self.semantic_dim, int(space_dim/2)),
            #     nn.Linear(int(space_dim/2), space_dim),
            # )
            s2e = nn.Linear(self.semantic_dim, space_dim)
            att = nn.Linear(self.voc.shape[1], space_dim)
            self.v2e_list.append(v2e)
            self.s2e_list.append(s2e)
            self.voc_att_list.append(att)
            if more_larger_space:
                base = base * 2


    def init_weights(self):
        if self.with_reg:
            nn.init.normal_(self.fc_reg.weight, 0, 0.001)
            nn.init.constant_(self.fc_reg.bias, 0)


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
    """
    comput the emsemble result of multispace
    """
    def conclude_multi_res(self, score_list):
        if self.num_space==1:
            return score_list[0]
        for i in range(self.num_space):
            if i==0:
                semantic_res = torch.max(score_list[0], score_list[1])
            elif i>0 and i+1<self.num_space:
                semantic_res = torch.max(semantic_res, score_list[i+1])
        return semantic_res

    @auto_fp16()
    def forward(self, x, validation_class=None, bg_vector=None):
        # with background vector generated
        if self.sync_bg:
            with torch.no_grad():
                self.vec[:, 0] = bg_vector
                self.vec_unseen[:,0] = bg_vector

        if self.with_avg_pool:
            x = self.avg_pool(x)
        x_box = x.view(x.size(0), -1)#512 256 7 7
        bbox_pred = self.fc_reg(x_box) if self.with_reg else None

        x_semantic = x.view(x.size(0), -1)
        for layer in self.FFN:
            x_semantic = self.relu(layer(x_semantic))
        compressed_feature = x_semantic  #(512,1024) 


        space_score_list = []
        feature_list = []
        semantic_list = []
        validation_score_list = []
        for i in range(self.num_space):
            v2e = self.v2e_list[i]
            s2e = self.s2e_list[i]
            att = self.voc_att_list[i]
            # 映射到子空间，计算与词表的相似度
            voc_emb = F.normalize(s2e(self.voc.t()).t(),2,0)
            vis_emb = v2e(compressed_feature)   #  return ?
            semantic_score = torch.mm(vis_emb, voc_emb)
            # 词表相似度映射回子空间
            rebulit_fea = att(semantic_score)    # or ?
            # 计算可见类相似度
            seen_emb = s2e(self.vec.t()).t()
            seen_emb_norm = seen_emb.clone()
            seen_emb_norm[:,1:] = F.normalize(seen_emb_norm[:,1:],2,0)
            space_score = torch.mm(rebulit_fea, seen_emb)
            # 保存结果
            space_score_list.append(space_score)
            semantic_list.append(seen_emb)  # contrastive loss require
            feature_list.append(rebulit_fea)  # contrastive loss require
            
            # 迁移损失部分
            # 1.提取训练类结果 经过softmax
            if validation_class is None:
                continue
            indices = torch.tensor([j for j in range(self.vec.shape[1]) if j!=validation_class]).cuda()
            train_res = torch.index_select(space_score, 1, indices)
            train_score = F.softmax(train_res, dim=1)
            # 2.emb空间 重构结果
            seen_train_emb = seen_emb[:, indices]
            rebuilt_train_emb = torch.mm(train_score, seen_train_emb.t())
            # 3.迁移类预测
            val_sem = seen_emb[:, validation_class].view(-1,1)
            validation_score = torch.mm(rebuilt_train_emb, val_sem)
            # 4.保存结果
            validation_score_list.append(validation_score)
        
        #compute final score
        res_score = self.conclude_multi_res(space_score_list)
        if validation_class is None:
            return res_score, bbox_pred
        return res_score, bbox_pred, semantic_list, feature_list, validation_score_list

    def get_target(self, sampling_results, gt_bboxes, gt_labels,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_seen_classes
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
             validation_class=None,
             validation_score_list=None,
             semantic_list=None,
             feature_list=None,
             reduction_override=None):
        losses = dict()
        # 分类损失
        if semantic_score is not None:
            avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            losses['loss_cls'] = self.loss_cls(
                semantic_score,
                labels,
                label_weights,
                avg_factor=avg_factor,
                reduction_override=reduction_override)
            losses['acc'] = accuracy(semantic_score, labels)
        # 回归损失
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
        # 对比损失
        if feature_list is not None:
            loss_con = 0
            loss_con_sem = 0
            for features, semantics in zip(feature_list, semantic_list):
                loss_con += self.loss_con(
                    features,
                    labels
                )
                loss_con_sem += self.loss_con_sem(
                    semantics,
                    features,
                    labels
                )
            if loss_con != 0:
                losses['loss_con'] = loss_con
            if loss_con_sem != 0:
                losses['loss_con_sem'] = loss_con_sem
        # 迁移损失 CE
        if validation_score_list is not None:
            val_index = labels == validation_class
            bin_lab = torch.zeros_like(labels)
            bin_lab[val_index] = 1
            loss_val = 0
            for val_score in validation_score_list:
                val_bin = val_score.repeat(1,2)
                val_bin[:,0] = 1 - val_bin[:,0]
                loss_val+=self.loss_val(
                val_bin,  # (512,1) --> (512,2)
                bin_lab
            )
            losses['loss_val'] = loss_val


        return losses


    def seen2unseen(self, scores, seen_vec, unseen_vec):
        seen_scores = torch.mm(scores, seen_vec.t())
        seen_scores = torch.mm(seen_scores, seen_vec)
        unseen_scores = torch.mm(scores, seen_vec.t())
        unseen_scores = torch.mm(unseen_scores, unseen_vec)
        return seen_scores, unseen_scores


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
        
        # select space to rebuild feature.
        # default: largest space
        # chooice: all space with max filter
        seen_bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                     self.target_stds, img_shape)
        unseen_bboxes = delta2bbox(rois[:, 1:], bbox_pred, self.target_means,
                                     self.target_stds, img_shape)

        if self.gzsd:
            if self.inference_multi:
                seen_scores_list = []
                unseen_scores_list = []
                for i in range(self.num_space):
                    seen_vec = self.s2e_list[i](self.vec.t()).t()
                    unseen_vec = self.s2e_list[i](self.vec_unseen.t()).t()
                    seen_vec[:,1:] = F.normalize(seen_vec[:,1:],2,0)
                    unseen_vec[:,1:] = F.normalize(unseen_vec[:,1:],2,0)

                    seen_scores, unseen_scores = self.seen2unseen(scores, seen_vec, unseen_vec)
                    seen_scores_list.append(seen_scores)
                    unseen_scores_list.append(unseen_scores)
                seen_scores = self.conclude_multi_res(seen_scores_list)
                unseen_scores = self.conclude_multi_res(unseen_scores_list)
            else:
                seen_vec = self.s2e_list[0](self.vec.t()).t()
                unseen_vec = self.s2e_list[0](self.vec_unseen.t()).t()
                seen_vec[:,1:] = F.normalize(seen_vec[:,1:],2,0)
                unseen_vec[:,1:] = F.normalize(unseen_vec[:,1:],2,0) 
                seen_scores, unseen_scores = self.seen2unseen(scores, seen_vec, unseen_vec)

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
            if self.inference_multi:
                seen_scores_list = []
                for i in range(self.num_space):
                    seen_vec = self.s2e_list[i](self.vec.t()).t()
                    unseen_vec = self.s2e_list[i](self.vec_unseen.t()).t()
                    seen_scores, _ = self.seen2unseen(scores, seen_vec, unseen_vec)
                    seen_scores_list.append(seen_scores)
                scores = self.conclude_multi_res(seen_scores_list)
            else:
                scores = torch.mm(scores, seen_vec.t())
                scores = torch.mm(scores, seen_vec)
        # TODO ZSD  open these lines when unseen inference
        if not self.seen_class:
            if self.inference_multi:
                unseen_scores_list = []
                for i in range(self.num_space):
                    seen_vec = self.s2e_list[i](self.vec.t()).t()
                    unseen_vec = self.s2e_list[i](self.vec_unseen.t()).t()
                    _, unseen_scores = self.seen2unseen(scores, seen_vec, unseen_vec)
                    unseen_scores_list.append(unseen_scores)
                scores = self.conclude_multi_res(unseen_scores_list)
            else:
                scores = torch.mm(scores, seen_vec.t())
                scores = torch.mm(scores, unseen_vec)

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
