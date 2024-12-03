# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .smt import SMT
from .convnext import ConvNeXt

from .adapter_swin_transformer import Adapter_SwinTransformer
from .adapter_convnext import Adapter_ConvNeXt
from .adapter_smt import Adapter_SMT

from .swin_transformer_diffusion_finetune import SwinTransformer_Diffusion_Finetune
from .convnext_diffusion_finetune import ConvNeXt_Diffusion_Finetune
from .smt_diffusion_finetune import SMT_Diffusion_Finetune

from .adapter_swin_transformer_diffusion_finetune import Adapter_SwinTransformer_Diffusion_Finetune
from .adapter_convnext_diffusion_finetune import Adapter_ConvNeXt_Diffusion_Finetune
from .adapter_smt_diffusion_finetune import Adapter_SMT_Diffusion_Finetune



def build_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE

    # accelerate layernorm
    if config.FUSED_LAYERNORM:
        try:
            import apex as amp
            layernorm = amp.normalization.FusedLayerNorm
        except:
            layernorm = None
            print("To use FusedLayerNorm, please install apex.")
    else:
        import torch.nn as nn
        layernorm = nn.LayerNorm


    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                norm_layer=layernorm,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                fused_window_process=config.FUSED_WINDOW_PROCESS)

    elif model_type == 'adapter_swin':
        model = Adapter_SwinTransformer(is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
                                     pretrain_image_size=config.MODEL.VCNU_SWIN.PRETRAIN_IMAGE_SIZE,
                                     model_style=config.MODEL.VCNU_SWIN.MODEL_STYLE,
                                     training_mode=config.MODEL.VCNU_SWIN.TRAINING_MODE,
                                     use_layerscales=config.MODEL.VCNU_SWIN.USE_LAYERSCALE,
                                     layer_scale_init_value=config.MODEL.VCNU_SWIN.LAYER_SCALE_INIT_VALUE,

                                     img_size=config.DATA.IMG_SIZE,
                                     patch_size=config.MODEL.VCNU_SWIN.PATCH_SIZE,
                                     # out_dim=config.MODEL.VCNU_SWIN.OUT_DIM,
                                     in_chans=config.MODEL.VCNU_SWIN.IN_CHANS,
                                     num_classes=config.MODEL.NUM_CLASSES,
                                     embed_dim=config.MODEL.VCNU_SWIN.EMBED_DIM,
                                     depths=config.MODEL.VCNU_SWIN.DEPTHS,
                                     num_heads=config.MODEL.VCNU_SWIN.NUM_HEADS,
                                     window_size=config.MODEL.VCNU_SWIN.WINDOW_SIZE,
                                     mlp_ratio=config.MODEL.VCNU_SWIN.MLP_RATIO,
                                     qkv_bias=config.MODEL.VCNU_SWIN.QKV_BIAS,
                                     qk_scale=config.MODEL.VCNU_SWIN.QK_SCALE,
                                     drop_rate=config.MODEL.DROP_RATE,
                                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                     ape=config.MODEL.VCNU_SWIN.APE,
                                     norm_layer=layernorm,
                                     patch_norm=config.MODEL.VCNU_SWIN.PATCH_NORM,
                                     use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                     fused_window_process=config.FUSED_WINDOW_PROCESS)
    elif model_type == 'smt':
        model = SMT(
            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.VCNU_SMT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.VCNU_SMT.EMBED_DIMS,
            ca_num_heads=config.MODEL.VCNU_SMT.CA_NUM_HEADS,
            sa_num_heads=config.MODEL.VCNU_SMT.SA_NUM_HEADS,
            mlp_ratios=config.MODEL.VCNU_SMT.MLP_RATIOS,
            qkv_bias=config.MODEL.VCNU_SMT.QKV_BIAS,
            qk_scale=config.MODEL.VCNU_SMT.QK_SCALE,
            use_layerscale=config.MODEL.VCNU_SMT.USE_LAYERSCALE,  #
            depths=config.MODEL.VCNU_SMT.DEPTHS,
            ca_attentions=config.MODEL.VCNU_SMT.CA_ATTENTIONS,
            head_conv=config.MODEL.VCNU_SMT.HEAD_CONV,
            expand_ratio=config.MODEL.VCNU_SMT.EXPAND_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    elif model_type == 'adapter_smt':
        model = Adapter_SMT(
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
            patch_size=config.MODEL.VCNU_SMT.PATCH_SIZE,
            model_style=config.MODEL.VCNU_SMT.MODEL_STYLE,

            use_layerscale=config.MODEL.VCNU_SMT.USE_LAYERSCALE,
            layerscale_value=config.MODEL.VCNU_SMT.LAYERSCALE_VALUE,  #

            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.VCNU_SMT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.VCNU_SMT.EMBED_DIMS,
            ca_num_heads=config.MODEL.VCNU_SMT.CA_NUM_HEADS,
            sa_num_heads=config.MODEL.VCNU_SMT.SA_NUM_HEADS,
            mlp_ratios=config.MODEL.VCNU_SMT.MLP_RATIOS,
            qkv_bias=config.MODEL.VCNU_SMT.QKV_BIAS,
            qk_scale=config.MODEL.VCNU_SMT.QK_SCALE,
            depths=config.MODEL.VCNU_SMT.DEPTHS,
            ca_attentions=config.MODEL.VCNU_SMT.CA_ATTENTIONS,
            head_conv=config.MODEL.VCNU_SMT.HEAD_CONV,
            expand_ratio=config.MODEL.VCNU_SMT.EXPAND_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    elif model_type == 'convnext':
        model = ConvNeXt(
            in_chans=config.MODEL.VCNU_CONVNEXT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VCNU_CONVNEXT.DEPTHS,
            dims=config.MODEL.VCNU_CONVNEXT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layer_scale_init_value=config.MODEL.VCNU_CONVNEXT.LAYER_SCALE_INIT_VALUE,
            )

    elif model_type == 'adapter_convnext':
        model = Adapter_ConvNeXt(
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
            patch_size=config.MODEL.VCNU_CONVNEXT.PATCH_SIZE,
            img_size=config.DATA.IMG_SIZE,
            model_style=config.MODEL.VCNU_CONVNEXT.MODEL_STYLE,

            in_chans=config.MODEL.VCNU_CONVNEXT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VCNU_CONVNEXT.DEPTHS,
            dims=config.MODEL.VCNU_CONVNEXT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layer_scale_init_value=config.MODEL.VCNU_CONVNEXT.LAYER_SCALE_INIT_VALUE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)
# --------------------------------------diffusion tuning -------------------------------------------
    elif model_type == 'swin_diffusion_finetune':
        model = SwinTransformer_Diffusion_Finetune(
            finetune_mode=config.TRAIN.FINETUNE_MODE,
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,

            img_size=config.DATA.IMG_SIZE,
            patch_size=config.MODEL.SWIN.PATCH_SIZE,
            in_chans=config.MODEL.SWIN.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dim=config.MODEL.SWIN.EMBED_DIM,
            depths=config.MODEL.SWIN.DEPTHS,
            num_heads=config.MODEL.SWIN.NUM_HEADS,
            window_size=config.MODEL.SWIN.WINDOW_SIZE,
            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
            qk_scale=config.MODEL.SWIN.QK_SCALE,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            ape=config.MODEL.SWIN.APE,
            norm_layer=layernorm,
            patch_norm=config.MODEL.SWIN.PATCH_NORM,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
            fused_window_process=config.FUSED_WINDOW_PROCESS)

    elif model_type == 'adapter_swin_diffusion_finetune':
        model = Adapter_SwinTransformer_Diffusion_Finetune(
             add_extra_adapter=config.MODEL.VCNU_SWIN.ADD_EXTRA_ADAPTER,
             finetune_mode=config.TRAIN.FINETUNE_MODE,
             is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
             pretrain_image_size=config.MODEL.VCNU_SWIN.PRETRAIN_IMAGE_SIZE,
             model_style=config.MODEL.VCNU_SWIN.MODEL_STYLE,
             training_mode=config.MODEL.VCNU_SWIN.TRAINING_MODE,
             use_layerscales=config.MODEL.VCNU_SWIN.USE_LAYERSCALE,
             layer_scale_init_value=config.MODEL.VCNU_SWIN.LAYER_SCALE_INIT_VALUE,

             img_size=config.DATA.IMG_SIZE,
             patch_size=config.MODEL.VCNU_SWIN.PATCH_SIZE,
             # out_dim=config.MODEL.VCNU_SWIN.OUT_DIM,
             in_chans=config.MODEL.VCNU_SWIN.IN_CHANS,
             num_classes=config.MODEL.NUM_CLASSES,
             embed_dim=config.MODEL.VCNU_SWIN.EMBED_DIM,
             depths=config.MODEL.VCNU_SWIN.DEPTHS,
             num_heads=config.MODEL.VCNU_SWIN.NUM_HEADS,
             window_size=config.MODEL.VCNU_SWIN.WINDOW_SIZE,
             mlp_ratio=config.MODEL.VCNU_SWIN.MLP_RATIO,
             qkv_bias=config.MODEL.VCNU_SWIN.QKV_BIAS,
             qk_scale=config.MODEL.VCNU_SWIN.QK_SCALE,
             drop_rate=config.MODEL.DROP_RATE,
             drop_path_rate=config.MODEL.DROP_PATH_RATE,
             ape=config.MODEL.VCNU_SWIN.APE,
             norm_layer=layernorm,
             patch_norm=config.MODEL.VCNU_SWIN.PATCH_NORM,
             use_checkpoint=config.TRAIN.USE_CHECKPOINT,
             fused_window_process=config.FUSED_WINDOW_PROCESS)

    elif model_type == 'convnext_diffusion_finetune':
        model = ConvNeXt_Diffusion_Finetune(
            finetune_mode=config.TRAIN.FINETUNE_MODE,
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,

            in_chans=config.MODEL.VCNU_CONVNEXT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VCNU_CONVNEXT.DEPTHS,
            dims=config.MODEL.VCNU_CONVNEXT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layer_scale_init_value=config.MODEL.VCNU_CONVNEXT.LAYER_SCALE_INIT_VALUE,
            )

    elif model_type == 'adapter_convnext_diffusion_finetune':
        model = Adapter_ConvNeXt_Diffusion_Finetune(
            add_extra_adapter=config.MODEL.VCNU_CONVNEXT.ADD_EXTRA_ADAPTER,
            finetune_mode=config.TRAIN.FINETUNE_MODE,
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,
            patch_size=config.MODEL.VCNU_CONVNEXT.PATCH_SIZE,
            img_size=config.DATA.IMG_SIZE,
            model_style=config.MODEL.VCNU_CONVNEXT.MODEL_STYLE,

            in_chans=config.MODEL.VCNU_CONVNEXT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            depths=config.MODEL.VCNU_CONVNEXT.DEPTHS,
            dims=config.MODEL.VCNU_CONVNEXT.DIMS,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            layer_scale_init_value=config.MODEL.VCNU_CONVNEXT.LAYER_SCALE_INIT_VALUE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    elif model_type == 'smt_diffusion_finetune':
        model = SMT_Diffusion_Finetune(
            finetune_mode=config.TRAIN.FINETUNE_MODE,
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,

            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.VCNU_SMT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.VCNU_SMT.EMBED_DIMS,
            ca_num_heads=config.MODEL.VCNU_SMT.CA_NUM_HEADS,
            sa_num_heads=config.MODEL.VCNU_SMT.SA_NUM_HEADS,
            mlp_ratios=config.MODEL.VCNU_SMT.MLP_RATIOS,
            qkv_bias=config.MODEL.VCNU_SMT.QKV_BIAS,
            qk_scale=config.MODEL.VCNU_SMT.QK_SCALE,
            use_layerscale=config.MODEL.VCNU_SMT.USE_LAYERSCALE,  #
            depths=config.MODEL.VCNU_SMT.DEPTHS,
            ca_attentions=config.MODEL.VCNU_SMT.CA_ATTENTIONS,
            head_conv=config.MODEL.VCNU_SMT.HEAD_CONV,
            expand_ratio=config.MODEL.VCNU_SMT.EXPAND_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    elif model_type == 'adapter_smt_diffusion_finetune':
        model = Adapter_SMT_Diffusion_Finetune(
            finetune_mode=config.TRAIN.FINETUNE_MODE,
            is_efficient_finetune=config.TRAIN.EFFICIENT_FINETUNE,

            patch_size=config.MODEL.VCNU_SMT.PATCH_SIZE,
            model_style=config.MODEL.VCNU_SMT.MODEL_STYLE,

            use_layerscale=config.MODEL.VCNU_SMT.USE_LAYERSCALE,
            layerscale_value=config.MODEL.VCNU_SMT.LAYERSCALE_VALUE,  #

            img_size=config.DATA.IMG_SIZE,
            in_chans=config.MODEL.VCNU_SMT.IN_CHANS,
            num_classes=config.MODEL.NUM_CLASSES,
            embed_dims=config.MODEL.VCNU_SMT.EMBED_DIMS,
            ca_num_heads=config.MODEL.VCNU_SMT.CA_NUM_HEADS,
            sa_num_heads=config.MODEL.VCNU_SMT.SA_NUM_HEADS,
            mlp_ratios=config.MODEL.VCNU_SMT.MLP_RATIOS,
            qkv_bias=config.MODEL.VCNU_SMT.QKV_BIAS,
            qk_scale=config.MODEL.VCNU_SMT.QK_SCALE,
            depths=config.MODEL.VCNU_SMT.DEPTHS,
            ca_attentions=config.MODEL.VCNU_SMT.CA_ATTENTIONS,
            head_conv=config.MODEL.VCNU_SMT.HEAD_CONV,
            expand_ratio=config.MODEL.VCNU_SMT.EXPAND_RATIO,
            drop_rate=config.MODEL.DROP_RATE,
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT)

    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
