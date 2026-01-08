# --------------------------------------------------------
# Diffusion Tuning
# Copyright (c) 2026 CAU
# Licensed under The MIT License [see LICENSE for details]
# Written by Guorun Li
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
import math
from timm.models.layers import to_2tuple

class Adapter(nn.Module):
    def __init__(self,
                 dim,
                 emb_dim,
                 proj_drop=0.,
                 model_style='conv',  # conv or trans
                 ):
        super().__init__()
        self.model_style = model_style
        self.dim = dim
        self.emb_dim = emb_dim
        self.activation = nn.GELU()
        self.down = nn.Conv2d(dim, emb_dim, 1, padding=0, bias=True) if self.model_style == 'conv' else nn.Linear(dim, emb_dim, bias=True)
        self.up = nn.Conv2d(emb_dim, dim, 1, bias=True) if self.model_style == 'conv' else nn.Linear(emb_dim, dim, bias=True)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # transformer style [B, L, C] or [B, H, W, C]
        # convolution style [B, C, H, W]
        x = self.down(x)
        x = self.up(self.activation(x))

        # return
        # trans style [B, L, C] or [B, H, W, C]
        # conv style [B,C,H,W]
        return self.proj_drop(x)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, emb_dim={self.emb_dim}, model_style={self.model_style}, "



