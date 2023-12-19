import logging

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.models.plugins import GeneralizedAttention
from mmdet.ops import ContextBlock, DeformConv, ModulatedDeformConv
from ..registry import BACKBONES
from ..utils import build_conv_layer, build_norm_layer
from ..VIT import ViT

class VitBackbone(nn.Module):

    def __init__(self,
                 image_size,
                 patch_size=1,
                 last_layer_dim=1,
                 feature_dims=1024,
                 ViT_depth=4,
                 heads=8,
                 channels=3,
                 dim_head=64,
                 dropout=0.1,
                 emb_dropout=0.1,
                 dcn=None,
                 gcb=None,
                 gen_attention=None):
        self.trans = ViT(image_size=image_size, 
                                    patch_size=patch_size, 
                                    num_classes=last_layer_dim, 
                                    dim=feature_dims, 
                                    depth=ViT_depth, 
                                    heads=heads, 
                                    mlp_dim=feature_dims, 
                                    pool='mean', 
                                    channels = channels, 
                                    dim_head = dim_head, 
                                    dropout = dropout, 
                                    emb_dropout = emb_dropout,
                                    with_mlp_head=False,
                                    learnable_pos_emd=True)


    def forward(self, x):

        out = self.trans(x)

        return out



