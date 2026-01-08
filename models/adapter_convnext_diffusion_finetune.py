# --------------------------------------------------------
# Diffusion Tuning
# Copyright (c) 2026 CAU
# Licensed under The MIT License [see LICENSE for details]
# Written by Guorun Li
# --------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
import math

import torch.utils.checkpoint as checkpoint
from .adapter import Adapter

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, qkv_bias=True, drop=0.,
                 model_style='trans', use_adapter=True,):
        super().__init__()
        self.use_adapter = use_adapter

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.adapter = Adapter(
            dim=dim,
            emb_dim=dim//4,
            proj_drop=drop,
            model_style=model_style,  # conv or trans
        ) if self.use_adapter else nn.Identity()

    def forward(self, x):
        # input (N, C, H, W)
        # memory (N, H, W, C)
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        perception_attn = self.pwconv2(x)
        # perception_attn = perception_attn.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        memory_attn = self.adapter(perception_attn)

        perception_attn = perception_attn.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        memory_attn = memory_attn.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        if self.gamma is not None:
            x = input + self.drop_path(self.gamma.unsqueeze(-1).unsqueeze(-1) * (perception_attn + memory_attn))
        else:
            x = input + self.drop_path(perception_attn + memory_attn)

        return x


class Adapter_ConvNeXt_Diffusion_Finetune(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, add_extra_adapter=True, finetune_mode='stage1', is_efficient_finetune=False, patch_size=4, img_size=224,
                 model_style='trans',
                 in_chans=3, num_classes=1000,
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1., use_checkpoint=False,
                 ):
        super().__init__()
        self.add_extra_adapter = add_extra_adapter
        self.use_checkpoint = use_checkpoint
        self.img_size = to_2tuple(img_size)
        patches_resolution = [self.img_size[0] // patch_size, self.img_size[1] // patch_size]

        # mode
        self.finetune_mode = finetune_mode
        self.is_efficient_finetune = is_efficient_finetune

        # step stage finetune, stage1 [False, True, False, False]
        # step cross layer1 [False, True, False, True]
        # sequence stage1 [True, True, False, False]
        # sequence cross layer [True, False, True, False]
        if not self.add_extra_adapter:
            if self.finetune_mode in ['stage0', 'stage1', 'stage2', 'stage3']:
                use_adapter_list = [False, False, False, False]
                index = int(self.finetune_mode.split('stage')[-1])
                for i in range(min(index + 1, len(use_adapter_list))):  
                    use_adapter_list[i] = True
            # step layer finetune, finetune layer-even, layer-odd full-finetune, respectively
            elif self.finetune_mode in ['part0', 'part1', ]:
                use_adapter_list = [None, None, None, None]
                if self.finetune_mode == 'part1':
                    use_adapter_list = [True, True, True, True, ]

            # sequence stage finetune, finetune stage0+embedding, stage0+embedding+stage1, stage0+embedding+stage1+stage2, full-finetune, respectively
            elif self.finetune_mode in ['sequence_stage0', 'sequence_stage1', 'sequence_stage2', ]:
                use_adapter_list = [False, False, False, False]
                index = int(self.finetune_mode.split('stage')[-1])
                for i in range(min(index + 1, len(use_adapter_list))):  
                    use_adapter_list[i] = True
            # step layer finetune, finetune layer-even, full-finetune, respectively
            elif self.finetune_mode in ['sequence_part0', ]:
                use_adapter_list = [None, None, None, None]
            else:
                use_adapter_list = [True, True, True, True, ]
        else:
            use_adapter_list = [True, True, True, True, ]

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        self.memory_downsampling = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            use_adapter = use_adapter_list[i]

            stage = nn.Sequential(
                *[Block(model_style=model_style,
                        use_adapter=((True if not is_odd(j) else False)
                                   if self.finetune_mode in ['sequence_part0', 'part0', ] else (True if is_odd(j) else False))
                                   if use_adapter is None else use_adapter,
                        dim=dims[i], drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)
        if self.is_efficient_finetune:
            # step stage finetune, finetune stage0+embedding, stage1, stage2, stage3, full-finetune, respectively
            if self.finetune_mode in ['stage0', 'stage1', 'stage2', 'stage3']:
                self.freeze_transferlearning_step_stage()
            # step layer finetune, finetune layer-even, layer-odd full-finetune, respectively
            elif self.finetune_mode in ['part0', 'part1', ]:
                self.freeze_transferlearning_step_cross()
            # sequence stage finetune, finetune stage0+embedding, stage0+embedding+stage1, stage0+embedding+stage1+stage2, full-finetune, respectively
            elif self.finetune_mode in ['sequence_stage0', 'sequence_stage1', 'sequence_stage2', ]:
                self.freeze_transferlearning_sequence_stage()
            # step layer finetune, finetune layer-even, full-finetune, respectively
            elif self.finetune_mode in ['sequence_part0', ]:
                self.freeze_transferlearning_sequence_cross()
            else:
                self.freeze_transferlearning()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                # nn.init.trunc_normal_(m.weight, std=.02)
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    # nn.init.constant_(m.bias, 0)
                    m.bias.data.zero_()

    def construct_memory(self, x):
        _, _, H, W = x.shape
        memorystream = self.uma(x)
        memorystream = memorystream.permute(0, 2, 3, 1).contiguous()

        assert memorystream.shape[
                   -1] == self.memory_dim, f"fast fourier transform error! pleace check fastFourierTrans() function. now the memory dim is {memorystream.shape[1]}, should be {self.memory_dim}!"
        return memorystream

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.stages[i], x)
            else:
                x = self.stages[i](x)

        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def freeze_transferlearning(self):
        for name, param in self.named_parameters():
            # print(name)
            if 'adapter' not in name:
                param.requires_grad = False

            if 'head' in name:
                param.requires_grad = True
            # ablation
            if name.startswith('norm'):
                param.requires_grad = True

        # checking code
        for name, param in self.named_parameters():
            print(f'Layer: {name}, Trainable: {param.requires_grad}')


    def freeze_transferlearning_step_stage(self):
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            else:
                if self.finetune_mode == 'stage0':
                    if 'stages.0' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

                elif self.finetune_mode == 'stage1':
                    if 'stages.1' not in name:
                        param.requires_grad = False
                elif self.finetune_mode == 'stage2':
                    if 'stages.2' not in name:
                        param.requires_grad = False
                elif self.finetune_mode == 'stage3':
                    if 'stages.3' not in name:
                        param.requires_grad = False
                else:
                    pass

            if 'head' in name:
                param.requires_grad = True

            # ablation
            if name.startswith('norm'):
                param.requires_grad = True

        # checking code
        for name, param in self.named_parameters():
            print(f'Layer: {name}, Trainable: {param.requires_grad}')

    def freeze_transferlearning_step_cross(self):
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            else:
                if self.finetune_mode == 'part0':
                    if 'stages.' in name:
                        name_list = name.split('.')
                        if not is_odd(int(name_list[2])):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                if self.finetune_mode == 'part1':
                    if 'stages.' in name:
                        name_list = name.split('.')
                        if not is_odd(int(name_list[2])):
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
                else:
                    pass

            if 'head' in name:
                param.requires_grad = True

            # ablation
            if name.startswith('norm'):
                param.requires_grad = True

        # checking code
        for name, param in self.named_parameters():
            print(f'Layer: {name}, Trainable: {param.requires_grad}')

    def freeze_transferlearning_sequence_stage(self):
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            else:
                if self.finetune_mode == 'sequence_stage0':
                    if 'stages.0' not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

                elif self.finetune_mode == 'sequence_stage1':
                    if 'stages.1' not in name:
                        param.requires_grad = False

                    if 'stages.0' in name:
                        param.requires_grad = True
                elif self.finetune_mode == 'sequence_stage2':
                    if 'stages.2' not in name:
                        param.requires_grad = False

                    if 'stages.0' in name:
                        param.requires_grad = True
                    elif 'stages.1' in name:
                        param.requires_grad = True
                    else: pass
                elif self.finetune_mode == 'sequence_stage3':
                    if 'stages.3' not in name:
                        param.requires_grad = False

                    if 'stages.0' in name:
                        param.requires_grad = True
                    elif 'stages.1' in name:
                        param.requires_grad = True
                    elif 'stages.2' in name:
                        param.requires_grad = True
                    else: pass
                else:
                    pass

            if 'head' in name:
                param.requires_grad = True

            # ablation
            if name.startswith('norm'):
                param.requires_grad = True

        # checking code
        for name, param in self.named_parameters():
            print(f'Layer: {name}, Trainable: {param.requires_grad}')

    def freeze_transferlearning_sequence_cross(self):
        for name, param in self.named_parameters():
            if 'adapter' not in name:
                param.requires_grad = False
            else:

                if self.finetune_mode == 'sequence_part0':
                    if 'stages.' in name:
                        name_list = name.split('.')
                        if not is_odd(int(name_list[3])):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False

                else:
                    pass

            if 'head' in name:
                param.requires_grad = True

            # ablation
            if name.startswith('norm'):
                param.requires_grad = True

        # checking code
        for name, param in self.named_parameters():
            print(f'Layer: {name}, Trainable: {param.requires_grad}')

def is_odd(number):
    return isinstance(number, int) and number % 2 != 0

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

@register_model
def adapter_convnext_tiny(pretrained=False,in_22k=False, **kwargs):
    model = Adapter_ConvNeXt_Diffusion_Finetune(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_tiny_22k'] if in_22k else model_urls['convnext_tiny_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def adapter_convnext_small(pretrained=False,in_22k=False, **kwargs):
    model = Adapter_ConvNeXt_Diffusion_Finetune(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
    if pretrained:
        url = model_urls['convnext_small_22k'] if in_22k else model_urls['convnext_small_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def adapter_convnext_base(pretrained=False, in_22k=False, **kwargs):
    model = Adapter_ConvNeXt_Diffusion_Finetune(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    if pretrained:
        url = model_urls['convnext_base_22k'] if in_22k else model_urls['convnext_base_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def adapter_convnext_large(pretrained=False, in_22k=False, **kwargs):
    model = Adapter_ConvNeXt_Diffusion_Finetune(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
    if pretrained:
        url = model_urls['convnext_large_22k'] if in_22k else model_urls['convnext_large_1k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def adapter_convnext_xlarge(pretrained=False, in_22k=False, **kwargs):
    model = Adapter_ConvNeXt_Diffusion_Finetune(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
    if pretrained:
        assert in_22k, "only ImageNet-22K pre-trained ConvNeXt-XL is available; please set in_22k=True"
        url = model_urls['convnext_xlarge_22k']
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


def test(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth

    #     FINETUNE_MODE
    # step stage finetune ['stage0', 'stage1', 'stage2', 'stage3' ], finetune stage0+embedding, stage1, stage2, stage3, full-finetune, respectively
    # step layer finetune ['part0', 'part1'], finetune layer-even, layer-odd full-finetune, respectively
    # sequence stage finetune [ 'sequence_stage0', 'sequence_stage1', 'sequence_stage2',], finetune stage0+embedding, stage0+embedding+stage1, stage0+embedding+stage1+stage2, full-finetune, respectively
    # step layer finetune [ 'sequence_part0',], finetune layer-even, full-finetune, respectively
    #  efficient finetune  -> 'fullfinetune'

    model = Adapter_ConvNeXt_Diffusion_Finetune(
        add_extra_adapter=True,
        finetune_mode='sequence_stage2',
        is_efficient_finetune=True,
        model_style='trans',
        patch_size=4, use_checkpoint=True,
        depths=[ 3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model


def count_gradients(model):
    # 计算需要计算梯度的参数量
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 计算总参数量
    total_params = sum(p.numel() for p in model.parameters())

    # 计算比值
    ratio = num_trainable_params / total_params

    return total_params, num_trainable_params, ratio

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from thop import profile
    from pytorch_model_summary import summary
    import time

    print(torch.__version__)
    # net = swinFocus_tiny_patch4_window7_224().cuda()
    net = test().cuda()
    import torchsummary

    # torchsummary.summary(net)
    print(net)
    image = torch.rand(1, 3, 224, 224).cuda()
    # time_step=torch.tensor([999] * 1, device="cuda")
    # f, p = get_model_complexity_info(net, image, as_strings=True, print_per_layer_stat=False, verbose=False)
    # f, p = profile(net, inputs=(image, time_step))

    f, p = profile(net, inputs=(image,))
    # f, p = summary(net, (image, time_step))
    print('flops:%f' % f)
    print('params:%f' % p)
    print('flops: %.1f G, params: %.1f M' % (f / 1e9, p / 1e6))

    s = time.time()
    with torch.no_grad():
        out = net(image, )

    print('infer_time:', time.time() - s)
    print("FPS:%f" % (1 / (time.time() - s)))

    # print(out.shape)

    total_params, num_trainable_params, ratio = count_gradients(net)
    print(f'total_params: {total_params / 1e6 : .2f} M')
    print(f'num_trainable_params: {num_trainable_params / 1e6 : .2f} M')
    print(f'ratio: {ratio * 100 : .2f} %')