import torch
import numpy as np
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def sincos_pos_emd(num_patches,dim):
    pe = torch.zeros(num_patches, dim) # 建立空表，每行代表一个词的位置，每列代表一个编码位
    position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1) # 建个arrange表示词的位置以便公式计算，size=(max_len,1)
    div_term = torch.exp(torch.tensor(torch.arange(0, dim, 2,dtype=torch.float) * 
                                                            -(math.log(10000.0) / float(dim)),dtype=torch.float))
    pe[:, 0::2] = torch.sin(position * div_term)  # 计算偶数维度的pe值
    pe[:, 1::2] = torch.cos(position * div_term)  # 计算奇数维度的pe值
    pe = pe.unsqueeze(0)  # size=(1, L, d_model)
    pe.requires_grad=False
    return pe.cuda()

# classes
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()
    def forward(self, x):
        return 0.5*x*(1+torch.tanh(np.sqrt(2/np.pi)*(x+0.044715*torch.pow(x,3))))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, head_dim = 64, dropout = 0.):
        super().__init__()
        inner_dim = head_dim *  heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TransformerEncoder(nn.Module):
    """
    dim: input dim and output dim
    depth: number of encoder in module
    heads: number of MHSA heads
    head_dim: output dim of each MHSA head
    mlp_dim: hidden dim of feedforward network 
    """
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, head_dim = head_dim, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout = 0.):
        super().__init__()
        self.encoder = TransformerEncoder(dim=dim, depth=depth,heads=heads, head_dim=head_dim, mlp_dim=mlp_dim,dropout=dropout)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., with_mlp_head=False, learnable_pos_emd=False):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width), # 512 256 7 7  -> 512 49 256
            nn.Linear(patch_dim, dim), # 256,300  -> 512 49 300  each pixel 300 channel
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  if learnable_pos_emd else sincos_pos_emd(num_patches + 1, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = TransformerEncoder(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.with_head = with_mlp_head
        if self.with_head:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape  # 512 49(7*7) 300

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b) # 512 1 300
        x = torch.cat((cls_tokens, x), dim=1) # 512 (n + 1) 300
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)  # directly return with out mlp_head   return x
        if self.with_head:
            return self.mlp_head(x)
        else:
            return x
