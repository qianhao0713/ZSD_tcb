from xml.dom.domreg import well_known_implementations
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..registry import LOSSES
from .utils import weighted_loss


@LOSSES.register_module
class ContrastiveSemLoss(nn.Module):
    def __init__(self, loss_weight, tal, foreground=False):
        super(ContrastiveSemLoss, self).__init__()
        self.loss_weight = loss_weight
        self.tal = tal
        self.foreground = foreground
        

    
    def forward(self, attribute_vec, feature, label):
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
        
        n, _ = fea.size() # d feature 维度  n 同一批前景feature个数
        _, s_n = attribute_vec.size() # s_n可见类语义向量个数


        value_all = torch.mm(attribute_vec.t(), fea.t()) # 66 x n
        fea_norm = torch.norm(fea, p=2, dim=1) #n
        att_norm = torch.norm(attribute_vec, p=2, dim=0) #66
        div_all = torch.mm(att_norm.view(-1,1), fea_norm.view(1,-1))
        ex_all = torch.div(value_all, self.tal *div_all)
        d_all =  torch.exp(ex_all)
        for idx in range(n):
            # 目标区域 以及正负样本生成 公式12
            anchor = fea[idx,:]
            an_lab = lab[idx]
            pos_val_sem = d_all[an_lab, idx]
            neg_val_sem = torch.sum(d_all[:,idx]) - pos_val_sem
            loss_sem_anc = -torch.log(pos_val_sem 
                                        / (pos_val_sem + neg_val_sem)
                                        )
            loss += loss_sem_anc
        # # 随机挑选  或者  遍历
        # for idx in range(n):
        #     # 目标区域 以及正负样本生成 公式12
        #     anchor = fea[idx,:]
        #     an_lab = lab[idx]
        #     # 目标语义
        #     loss_sem_anc = 0
        #     value = torch.mv(attribute_vec.t(),anchor) #66 *1
        #     anchor_norm = torch.norm(anchor)
        #     att_norm = torch.norm(attribute_vec, p=2, dim=0)
        #     ex = torch.div(value, self.tal * anchor_norm * att_norm)
        #     d = torch.exp(ex)
        #     pos_val_sem = d[an_lab]
        #     neg_val_sem = torch.sum(d) - pos_val_sem
        #     loss_sem_anc = -torch.log(pos_val_sem 
        #                                 / (pos_val_sem + neg_val_sem)
        #                                 )
        #     loss_n += loss_sem_anc
        loss_con = self.loss_weight * loss / n
        return loss_con
