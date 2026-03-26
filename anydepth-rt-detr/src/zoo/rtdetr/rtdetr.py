"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
    def forward(self, x, targets=None, backbone_skip=[False, False, False, False], encoder_skip=[False,], decoder_depth=6):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        
        x_backbone = self.backbone(x, skip=backbone_skip)
        # print("output of backbone:", len(x_backbone))
        # for i in range(len(x_backbone)):
        #     print(f"shape of output[{i}]: {x_backbone[i].shape}")

        x_encoder = self.encoder(x_backbone, skip=encoder_skip)        
        # print("output of encoder:")
        # for i in range(len(x_encoder)):
        #     print(f"shape of output[{i}]: {x_encoder[i].shape}")

        x = self.decoder(x_encoder, depth=decoder_depth, targets=targets)
        # if self.training:
        x['intermediate_backbone'] = x_backbone
        x['intermediate_encoder'] = x_encoder

        return x
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
