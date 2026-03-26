'''by lyuwenyu
'''

import copy
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from .utils import get_activation

from src.core import register


__all__ = ['HybridEncoder']



class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# woochul
class SkippableConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None, grouped=False, skippable=False):
        super().__init__()
        if grouped:  # exp: 2025.05.22 for channel-wise conv
            self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias,
            groups=ch_out)
        else:
            self.conv = nn.Conv2d(
                ch_in, 
                ch_out, 
                kernel_size, 
                stride, 
                padding=(kernel_size-1)//2 if padding is None else padding, 
                bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        # woochul
        self.skippable = skippable
        if self.skippable == False:  # mandatory layers need switchable BN
            self.norm_skippable = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x, skip=False):
        x = self.conv(x)
        if self.skippable == False and skip == True:
            # print(f"norm_skip")
            x = self.norm_skippable(x)
        else: 
            # print("norm")
            x = self.norm(x)
        return self.act(x) 

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu', skippable=False):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        
        # orig
        self.conv1 = SkippableConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None, skippable=skippable)
        self.conv2 = SkippableConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None, skippable=skippable)

        # wchkang: exp option #2
        # self.conv1 = SkippableConvNormLayer(ch_in, ch_in, 3, 1, padding=1, act='relu', skippable=skippable)
        # self.conv2 = SkippableConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None, skippable=skippable)
        
        # wchkang: exp option #2
        # expansion_ratio = 2 # 4
        # self.conv1 = SkippableConvNormLayer(ch_in, ch_in * expansion_ratio, 1, 1, padding=0, act='relu', skippable=skippable)
        # self.conv2 = SkippableConvNormLayer(ch_in * expansion_ratio, ch_in * expansion_ratio, 3, 1, padding=1, act='relu', grouped=True, skippable=skippable)
        # self.conv3 = SkippableConvNormLayer(ch_in * expansion_ratio, ch_out, 1, 1, padding=0, act=None, skippable=skippable)
        
        self.act = nn.Identity() if act is None else get_activation(act) 
        self.skippable = skippable

    def forward(self, x, skip=False):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x, skip=skip) + self.conv2(x, skip=skip) # orig

            # wchkang: similar to bottleneck block
            # y = self.conv1(x, skip=skip)
            # y = self.conv2(y, skip=skip)

            # wchkang: similar to bottleneck block
            # y = self.conv1(x, skip=skip)
            # y = self.conv2(y, skip=skip)
            # y = self.conv3(y, skip=skip)
            
        return self.act(y) # orig
        # return self.act(y + x) # wchkang: add shortcut connection like ResNet

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std


# woochul ...
class SkippableSequentialBlocks(nn.Sequential):
    """Skips some blocks in the stage"""
    def forward(self, input, skip = False):
        """Extends nn.Sequential's forward for skipping some blocks
        Args:
            x (Tensor): input tensor
            skip (bool): if True, skip the last half blocks in the stage.
        """
        for i in range(len(self)):
            if self[i].skippable == True and skip == True:
                # print(f"Skip block {i}")
                pass
            # elif self[i].skippable == True and skip == False:  # exp: 2025.05.18
            #     # print(f"Run block {i} with residual connection")
            #     input = self[i](input, skip) + input 
            else:
                # print(f"Run block {i}")
                input = self[i](input, skip)
                
        return input

# woochul
class SkippableInputProjection(nn.Module):
    def __init__(self, in_channel, hidden_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.norm_skippable = nn.BatchNorm2d(hidden_dim)

    def forward(self, x, skip=False):
        x = self.conv(x)
        if skip == True:
            # print(f"input projection: norm_skippable")
            x = self.norm_skippable(x)
        else:
            # print(f"input projection: norm")
            x = self.norm(x)
        return x

class CSPRepLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = SkippableConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act, skippable=False)
        self.conv2 = SkippableConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act, skippable=False)
        # original
        # self.bottlenecks = nn.Sequential(*[
        #     RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        # ])

        # woochul: 1 of 3 is shared blocks
        num_shared_blocks = (num_blocks  + 1 )  // 2 - 1

        # woochul exp: 2 of 3 are shared blocks
        # num_shared_blocks = (num_blocks + 1)  // 2 

        blocks = []
        for i in range(num_blocks):
            blocks.append(RepVggBlock(hidden_channels, hidden_channels, act=act, skippable=(i >= num_shared_blocks)))
        self.bottlenecks = SkippableSequentialBlocks(*blocks)

        if hidden_channels != out_channels:
            self.conv3 = SkippableConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act, skip=False)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x, skip=False):
        x_1 = self.conv1(x, skip=skip)
        x_1 = self.bottlenecks(x_1, skip=skip)
        x_2 = self.conv2(x, skip=skip)
        return self.conv3(x_1 + x_2) # orig
        # return self.conv3(x_1) # woochul: exp 

# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False,
                 ):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # woochul 
        self.norm1_skippable = nn.LayerNorm(d_model)
        self.norm2_skippable = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None, skip=False) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1_skippable(src) if skip else self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1_skippable(src) if skip else self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2_skippable(src) if skip else self.norm2(src)

        # if skip:
        #     pass
        #     print("attention norm_skippable")
        # else:
        #     pass
        #     print("attention norm")
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None, skip=False) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed, skip=skip)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_spatial_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_spatial_size = eval_spatial_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                # nn.Sequential(
                #     nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                #     nn.BatchNorm2d(hidden_dim)
                # )
                SkippableInputProjection(in_channel, hidden_dim)
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(SkippableConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act, skippable=False))
            self.fpn_blocks.append(
                # woochul. exp: use depth 2 instead of 3 (orig)
                # orig
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                # depth = 4
                # CSPRepLayer(hidden_dim * 2, hidden_dim, round(5 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                SkippableConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act, skippable=False)
            )
            self.pan_blocks.append(
                # woochul. exp: 
                # orig: depth=3
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
                # depth = 4
                # CSPRepLayer(hidden_dim * 2, hidden_dim, round(5 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_spatial_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_spatial_size[1] // stride, self.eval_spatial_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats, skip=[False,]):
        assert len(feats) == len(self.in_channels)
        
        proj_feats = [self.input_proj[i](feat, skip=skip[0]) for i, feat in enumerate(feats)]
        
        # woochul
        # if skip is not None:
        #     print("encoder skip:", skip)


        # encoder
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                # print(f"[woochul] encoder layer idx {i}")
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_spatial_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed, skip=skip[0])
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])

        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            # print(f"[woochul] encoder fpn_blocks idx: {idx}")
            feat_high = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_high = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_high, skip=skip[0])
            inner_outs[0] = feat_high
            upsample_feat = F.interpolate(feat_high, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[len(self.in_channels)-1-idx](torch.concat([upsample_feat, feat_low], dim=1), skip=skip[0])
            inner_outs.insert(0, inner_out)

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            # print(f"[woochul] encoder pan_blocks idx: {idx}")
            feat_low = outs[-1]
            feat_high = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low, skip=skip[0])
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_high], dim=1), skip=skip[0])
            outs.append(out)

        return outs
