from xml.dom.domreg import well_known_implementations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import LOSSES
from .utils import weighted_loss


@LOSSES.register_module
class ContrastiveLoss(nn.Module):
    def __init__(self, loss_weight, tal, foreground=False):
        super(ContrastiveLoss, self).__init__()
        self.loss_weight = loss_weight
        self.tal = tal
        self.foreground = foreground
    
    def forward(self, feature, label):
        """
        attribute_vec : seen and unseen class word vec
        feature : a batch bbox feature  只做前景
        label : a batch bbox ground label label>0
        """
        loss = 0
        # 挑选前景
        if self.foreground:
            index = label > 0
            fea = feature[index, :]
            lab = label[index]
        else:# 前景背景 均参与计算
            index = label > -1
            fea = feature[0:100, :]
            lab = label[0:100]
        
        n, _ = fea.size()
        value_all = torch.mm(fea, fea.t())
        fea_norm = torch.norm(fea, p=2, dim=1)
        div_all = torch.mm(fea_norm.view(-1,1), fea_norm.view(1,-1))
        ex_all = torch.div(value_all, self.tal *div_all)
        d_all =  torch.exp(ex_all)
        # 随机挑选  或者  遍历
        for idx in range(n):
            # 目标区域 以及正负样本生成 公式12
            anchor = fea[idx,:]
            an_lab = lab[idx]
            pos_idx = lab == an_lab
            neg_idx = lab != an_lab
             # 正例个数 至少有一个 不包含自身
            n_pos = sum(pos_idx) # sum
            pos_val = torch.sum(d_all[pos_idx, idx])
            neg_val = torch.sum(d_all[neg_idx, idx])
            pos_d = d_all[pos_idx, idx]
            loss_pos = -torch.log(pos_d/(pos_val+neg_val))
            loss_anc = torch.sum(loss_pos) / n_pos

            loss += loss_anc

        loss_con = self.loss_weight * loss / n
        return loss_con
