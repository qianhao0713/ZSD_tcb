import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..registry import LOSSES
from .utils import weight_reduce_loss


'''
require network output all class's similarity
no need sigmoid for binary_cross_entropy built within
81 class 0--background
'''
@LOSSES.register_module
class DiscriminstiveContrastiveLoss(nn.Module):
    def __init__(self,
                 contrastive_weight,
                 seen_unseen_similarity = None,# path of the file stored
                 reduction='mean',
                 loss_weight=1.0):
        super(DiscriminstiveContrastiveLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.alpha = contrastive_weight
        #  
        self.bce = nn.BCELoss().cuda()

    def forward(self,
                cls_score,
                label,
                seen_unseen_similarity,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        # input similarity
        self.seen2unseen = seen_unseen_similarity
        self.S, self.U = self.seen2unseen.shape

        loss_cls = self.loss_weight * self.binary_cross_entropy(
            cls_score[:, :self.S+1],
            label,
            unseen=False,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)

        unseen_score = cls_score[:,self.S+1:]
        loss_contrastive = self.loss_weight * self.binary_cross_entropy(
            unseen_score,
            label,
            unseen=True,
            weight=weight,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cls + self.alpha * loss_contrastive
    
    def _expand_seen2unseen_label(self, labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0).float()
        inds = torch.nonzero(labels >= 1).squeeze()
        if inds.numel() > 1:
            for id in inds:
                bin_labels[id,:] = self.seen2unseen[labels[id]-1,:]
            # idx = labels[inds] - 1
            # bin_labels[inds, :] = self.seen2unseen[idx,:]
        elif inds.numel() > 0:
            bin_labels[inds,:] = self.seen2unseen[labels[inds]-1,:]
        if label_weights is None:
            bin_label_weights = None
        else:
            bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)
        return bin_labels, bin_label_weights


    def _expand_binary_labels(self, labels, label_weights, label_channels):
        bin_labels = labels.new_full((labels.size(0), label_channels), 0)
        inds = torch.nonzero(labels >= 1).squeeze()
        if inds.numel() > 0:
            bin_labels[inds, labels[inds] - 1] = 1
        if label_weights is None:
            bin_label_weights = None
        else:
            bin_label_weights = label_weights.view(-1, 1).expand(
                label_weights.size(0), label_channels)
        return bin_labels, bin_label_weights


    #网络层输出不用添加sigmoid  
    def binary_cross_entropy(self,
                         pred,
                         label,
                         unseen=False,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):
        if unseen:
            label, weight = self._expand_seen2unseen_label(label, weight, pred.size(-1))
        else:
            if pred.dim() != label.dim():
                label, weight = self._expand_binary_labels(label, weight, pred.size(-1))

        # weighted element-wise losses
        if weight is not None:
            weight = weight.float()
        
        loss=self.bce(pred, label.float()) 
        # loss = F.binary_cross_entropy_with_logits(
        #     pred, label.float(), weight, reduction='none')
        # #do the reduction for the weighted loss
        # loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

        return loss