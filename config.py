# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import torch
import yaml
from yacs.config import CfgNode as CN

# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'  # asianface
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 32

# [SimMIM] Mask patch size for MaskGenerator
_C.DATA.MASK_PATCH_SIZE = 32
# [SimMIM] Mask ratio for MaskGenerator
_C.DATA.MASK_RATIO = 0.6
# use additional noise testing
# 'norm' -> no noise , support noice list ['gaussian', 'salt_pepper', poisson, laplacian, speckle, ]
# and mixed -> input['gaussian', 'poisson']
_C.DATA.NOISE_MODEL = 'norm'
_C.DATA.NOICE_MEAN = 0.  # gaussian noice and speckle noice mean
_C.DATA.NOICE_STD = 1.   # gaussian noice std speckle noice    guassian_noice=randn[]*std + mean, speckle_noice=img+img*gaussian
_C.DATA.NOICE_AMOUNT = 0.05  # salt_pepper noice amount     0.05->5%
_C.DATA.NOICE_LOC = 0.  # laplacian noice mean
_C.DATA.NOICE_SCALE = 1.  # laplacian noice scale
_C.DATA.NOICE_TYPES = ['gaussian', 'poisson']  # mixed noice

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_b_22kto1k_fft'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.15
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# generalVCNUs parameters
_C.MODEL.generalVCNUS = CN()
_C.MODEL.generalVCNUS.PRETRAIN_IMAGE_SIZE = 224
_C.MODEL.generalVCNUS.FILTER_STRATEGY1 = 12
_C.MODEL.generalVCNUS.FILTER_STRATEGY2 = 4
_C.MODEL.generalVCNUS.USE_FIBONACCI = True
_C.MODEL.generalVCNUS.PATCH_SIZE = 4
_C.MODEL.generalVCNUS.EMBED_CONV = 7
_C.MODEL.generalVCNUS.IN_CHANS = 3
_C.MODEL.generalVCNUS.EMBED_DIM = 64
_C.MODEL.generalVCNUS.DEPTHS = [3, 3, 12, 3]
_C.MODEL.generalVCNUS.RECE_FIELD = 7
_C.MODEL.generalVCNUS.KERNAL_SIZE = 11
_C.MODEL.generalVCNUS.MLP_RATIO = 4.
_C.MODEL.generalVCNUS.USE_BIAS = True
_C.MODEL.generalVCNUS.USE_LAYERSCALE = False
_C.MODEL.generalVCNUS.LAYERSCALE_VALUE = 1e-6
_C.MODEL.generalVCNUS.APE = False
_C.MODEL.generalVCNUS.PATCH_NORM = True
_C.MODEL.generalVCNUS.SAVE_FREQ = 30

_C.MODEL.generalVCNUS.ABLATION_STRATEGY = 'UMA'
_C.MODEL.generalVCNUS.USE_SEQUENCEFUNC = 'statistic'
_C.MODEL.generalVCNUS.AB_USE_SEQUENCEFUNC = 'UMA'  # statistic  fibonacci , linear
_C.MODEL.generalVCNUS.AB_STRATEGY = 'statistic'  # strategy1, strategy2, UMA

_C.MODEL.generalVCNUS.NUM_SCALE = 4
_C.MODEL.generalVCNUS.AB_WM = 'l'  # l->Linear conv -> Conv3
_C.MODEL.generalVCNUS.AB_AGGREGATION_ATTN = 'cat' # add, hard, cat
_C.MODEL.generalVCNUS.AB_AGGREGATION_LTM = 'add'  # add, hard, cat
_C.MODEL.generalVCNUS.AB_NORM_ATTN = True
_C.MODEL.generalVCNUS.AB_NORM_LTM = True
_C.MODEL.generalVCNUS.AB_NORM_ATTN_NAME = 'BN'  # LN->LayerNorm, BN->BatchNorm
_C.MODEL.generalVCNUS.AB_NORM_LTM_NAME = 'BN'  # LN->LayerNorm, BN->BatchNorm
_C.MODEL.generalVCNUS.AB_DOWNSAMPLING_STRATEGY = 'max'  # avg, conv, max
_C.MODEL.generalVCNUS.AB_PATCH_NORM_NAME = 'BN'  # LN->LayerNorm, BN->BatchNorm
_C.MODEL.generalVCNUS.AB_MEMORY_CREATION_STRATEGY = 'UMA'  # embedding param, UMA
_C.MODEL.generalVCNUS.MODEL_STYLE = 'trans'  #  conv or trans

# VCNU CONVNEXT parameters
_C.MODEL.VCNU_CONVNEXT = CN()
_C.MODEL.VCNU_CONVNEXT.NUM_SCALE = 4  # multi-conv parameter
_C.MODEL.VCNU_CONVNEXT.FILTER_STRATEGY1 = 18
_C.MODEL.VCNU_CONVNEXT.FILTER_STRATEGY2 = 6
_C.MODEL.VCNU_CONVNEXT.PRETRAIN_IMAGE_SIZE = 224
_C.MODEL.VCNU_CONVNEXT.PATCH_SIZE = 4
_C.MODEL.VCNU_CONVNEXT.OUT_DIM = None
_C.MODEL.VCNU_CONVNEXT.IN_CHANS = 3
_C.MODEL.VCNU_CONVNEXT.DIMS = [96, 192, 384, 768]
_C.MODEL.VCNU_CONVNEXT.QKV_BIAS = True
_C.MODEL.VCNU_CONVNEXT.DEPTHS = [ 3, 3, 9, 3]
_C.MODEL.VCNU_CONVNEXT.LAYER_SCALE_INIT_VALUE = 1e-6

_C.MODEL.VCNU_CONVNEXT.AB_NORM_ATTN = True
_C.MODEL.VCNU_CONVNEXT.AB_NORM_LTM = False
_C.MODEL.VCNU_CONVNEXT.MODEL_STYLE = 'trans'   # conv or trans
_C.MODEL.VCNU_CONVNEXT.USE_MEMORY_EMBEDDING = False  # lora style

_C.MODEL.VCNU_CONVNEXT.FINETUNE_MODE = 'stage0'
# step stage finetune ['stage0', 'stage1', 'stage2', 'stage3' ], finetune stage0+embedding, stage1, stage2, stage3, full-finetune, respectively
# step layer finetune ['part0', 'part1'], finetune layer-even, layer-odd full-finetune, respectively
# sequence stage finetune [ 'sequence_stage0', 'sequence_stage1', 'sequence_stage2',], finetune stage0+embedding, stage0+embedding+stage1, stage0+embedding+stage1+stage2, full-finetune, respectively
# step layer finetune [ 'sequence_part0',], finetune layer-even, full-finetune, respectively
#  sequence_stage0 == stage0, sequence_part0 == part0
_C.MODEL.VCNU_CONVNEXT.ADD_EXTRA_ADAPTER = True

# VCNU SMT parameters
_C.MODEL.VCNU_SMT = CN()
_C.MODEL.VCNU_SMT.NUM_SCALE = 4  # multi-conv parameter
_C.MODEL.VCNU_SMT.FILTER_STRATEGY1 = 23
_C.MODEL.VCNU_SMT.FILTER_STRATEGY2 = 7
_C.MODEL.VCNU_SMT.PRETRAIN_IMAGE_SIZE = 224
_C.MODEL.VCNU_SMT.PATCH_SIZE = 4
_C.MODEL.VCNU_SMT.OUT_DIM = None
_C.MODEL.VCNU_SMT.IN_CHANS = 3
_C.MODEL.VCNU_SMT.EMBED_DIMS = [64, 128, 256, 512]
_C.MODEL.VCNU_SMT.CA_NUM_HEADS = [4, 4, 4, -1]
_C.MODEL.VCNU_SMT.SA_NUM_HEADS = [-1, -1, 8, 16]
_C.MODEL.VCNU_SMT.MLP_RATIOS = [8, 6, 4, 2]
_C.MODEL.VCNU_SMT.QKV_BIAS = True
_C.MODEL.VCNU_SMT.QK_SCALE = None
_C.MODEL.VCNU_SMT.DEPTHS = [ 2, 2, 8, 1 ]
_C.MODEL.VCNU_SMT.CA_ATTENTIONS = [ 1, 1, 1, 0 ]
_C.MODEL.VCNU_SMT.HEAD_CONV = 3
_C.MODEL.VCNU_SMT.NUM_STAGES = 4
_C.MODEL.VCNU_SMT.EXPAND_RATIO = 2

_C.MODEL.VCNU_SMT.AB_NORM_ATTN = True
_C.MODEL.VCNU_SMT.AB_NORM_LTM = False
_C.MODEL.VCNU_SMT.MODEL_STYLE = 'trans'
_C.MODEL.VCNU_SMT.USE_LAYERSCALE = False
_C.MODEL.VCNU_SMT.LAYERSCALE_VALUE = 1e-4

_C.MODEL.VCNU_SMT.ADD_EXTRA_ADAPTER = True
_C.MODEL.VCNU_SMT.FINETUNE_MODE = 'full'
_C.MODEL.VCNU_SMT.USE_MEMORY_EMBEDDING = False  # lora style
# VCNU Swin Transformer parameters
_C.MODEL.VCNU_SWIN = CN()
_C.MODEL.VCNU_SWIN.NUM_SCALE = 4
_C.MODEL.VCNU_SWIN.FILTER_STRATEGY1 = 23
_C.MODEL.VCNU_SWIN.FILTER_STRATEGY2 = 7
_C.MODEL.VCNU_SWIN.PRETRAIN_IMAGE_SIZE = 224
_C.MODEL.VCNU_SWIN.AB_NORM_ATTN = True
_C.MODEL.VCNU_SWIN.AB_NORM_LTM = False
_C.MODEL.VCNU_SWIN.MODEL_STYLE = 'trans'   # conv or trans
_C.MODEL.VCNU_SWIN.TRAINING_MODE = 'tfs'  # tfs, finetune, efficient_ft
_C.MODEL.VCNU_SWIN.USE_LAYERSCALE = False  # use layerscale
_C.MODEL.VCNU_SWIN.LAYER_SCALE_INIT_VALUE = 1e-6  # init_value
_C.MODEL.VCNU_SWIN.FINETUNE_MODE = 'full'  # diffusion finetune -> [ part1, part2, full ]
_C.MODEL.VCNU_SWIN.ADD_EXTRA_ADAPTER = True  # diffusion finetune -> [ part1, part2, full ]

_C.MODEL.VCNU_SWIN.PATCH_SIZE = 4
_C.MODEL.VCNU_SWIN.OUT_DIM = None
_C.MODEL.VCNU_SWIN.IN_CHANS = 3
_C.MODEL.VCNU_SWIN.EMBED_DIM = 84
_C.MODEL.VCNU_SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.VCNU_SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.VCNU_SWIN.WINDOW_SIZE = 7
_C.MODEL.VCNU_SWIN.MLP_RATIO = 4.
_C.MODEL.VCNU_SWIN.QKV_BIAS = True
_C.MODEL.VCNU_SWIN.QK_SCALE = None
_C.MODEL.VCNU_SWIN.APE = False
_C.MODEL.VCNU_SWIN.PATCH_NORM = True
_C.MODEL.VCNU_SWIN.USE_MEMORY_EMBEDDING = False  # lora style


# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.FINETUNE_MODE = 'stage0'  # diffusion finetune
# step stage finetune ['stage0', 'stage1', 'stage2', 'stage3' ], finetune stage0+embedding, stage1, stage2, stage3, full-finetune, respectively
# step layer finetune ['part0', 'part1'], finetune layer-even, layer-odd full-finetune, respectively
# sequence stage finetune [ 'sequence_stage0', 'sequence_stage1', 'sequence_stage2',], finetune stage0+embedding, stage0+embedding+stage1, stage0+embedding+stage1+stage2, full-finetune, respectively
# step layer finetune [ 'sequence_part0',], finetune layer-even, full-finetune, respectively
#  sequence_stage0 == stage0, sequence_part0 == part0
_C.MODEL.SWIN.ADD_EXTRA_ADAPTER = True

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]

# Resnet
_C.MODEL.RESNET = CN()
_C.MODEL.RESNET.LAYERS = 4
# ViT
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCHES = 16
_C.MODEL.VIT.EMBED_DIM = 768
_C.MODEL.VIT.FEEDFORWARD_DIM = 3072
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.NUM_LAYERS = 12
_C.MODEL.VIT.ATTENTION_DROPOUT_RATE = 0.0
_C.MODEL.VIT.DROPOUT_RATE = 0.1
_C.MODEL.VIT.REPRESENTATION_SIZE = 768
_C.MODEL.VIT.LOAD_REPR_LAYER = False
_C.MODEL.VIT.CLASSIFIER = 'token'
_C.MODEL.VIT.POSITIONAL_EMBEDDING = '1d'
_C.MODEL.VIT.IN_CHANS = 3
# mobilevit
_C.MODEL.MOBILEVIT = CN()
_C.MODEL.MOBILEVIT.TYPE = 's'  # s xs xxs
# pvt
_C.MODEL.PYT = CN()
_C.MODEL.PYT.PATCH_SIZE = 4
_C.MODEL.PYT.IN_CHANS = 3
_C.MODEL.PYT.EMBED_DIMS = [32, 64, 160, 256]
_C.MODEL.PYT.NUM_HEADS = [1, 2, 5, 8]
_C.MODEL.PYT.MLP_RATIOS = [8, 8, 4, 4]
_C.MODEL.PYT.QKV_BIAS = True
_C.MODEL.PYT.QK_SCALE = None
_C.MODEL.PYT.DROP_RATE = 0.
_C.MODEL.PYT.ATTN_DROP_RATE = 0.
_C.MODEL.PYT.DROP_PATH_RATE = 0.
_C.MODEL.PYT.DEPTHS = [2, 2, 2, 2]
_C.MODEL.PYT.SR_RATIOS = [8, 4, 2, 1]
_C.MODEL.PYT.NUM_STAGES = 4
_C.MODEL.PYT.LINEAR = False
# efficientnetv2
_C.MODEL.EFFICIENTNET_V2 = CN()
_C.MODEL.EFFICIENTNET_V2.NAME = 'efficientnet_v2_s' # efficientnet_v2_s, efficientnet_v2_m, efficientnet_v2_l, efficientnet_v2_s_in21k
_C.MODEL.EFFICIENTNET_V2.DROPOUT = 0.1
_C.MODEL.EFFICIENTNET_V2.STOCHASTIC_DEPTH = 0.1


# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20  # vcnu 5
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4  # convnext 4e-3 vcnu 1e-3
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6  # convnext 1e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# Whether to use 'gamma' to  adjust  compir loss function:
# loss = CE(y,  y_hat) + gamma * E[||gaussian_disturbance - gaussian_disturbance_hat||^2 ]
_C.TRAIN.GAUSSIAN_GAMMA = 1.
# use computational irregular function to training model
_C.TRAIN.COMPIR_TRAIN = True

# Whether to use efficient finetuning to verify memory function
_C.TRAIN.EFFICIENT_FINETUNE = False
# diffusion tuning mode, support [dt1, dt2, dt3, dt4]
_C.TRAIN.FINETUNE_MODE = 'dt1'


# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1
# warmup_prefix used in CosineLRScheduler
_C.TRAIN.LR_SCHEDULER.WARMUP_PREFIX = True
# [SimMIM] Gamma / Multi steps value, used in MultiStepLRScheduler
_C.TRAIN.LR_SCHEDULER.GAMMA = 0.1
_C.TRAIN.LR_SCHEDULER.MULTISTEPS = []

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# [SimMIM] Layer decay for fine-tuning
_C.TRAIN.LAYER_DECAY = 1.0

# MoE
_C.TRAIN.MOE = CN()
# Only save model on master device
_C.TRAIN.MOE.SAVE_MASTER = False
# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8  # vcnu 0.2
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False
_C.TEST.SHUFFLE = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# [SimMIM] Whether to enable pytorch amp, overwritten by command line argument
_C.ENABLE_AMP = False

# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 30
# Frequency to logging info
_C.PRINT_FREQ = 100
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# for acceleration
_C.FUSED_WINDOW_PROCESS = False
_C.FUSED_LAYERNORM = False


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    def _check_args(name):
        if hasattr(args, name) and eval(f'args.{name}'):
            return True
        return False

    # merge from specific arguments
    if _check_args('batch_size'):
        config.DATA.BATCH_SIZE = args.batch_size
    if _check_args('data_path'):
        config.DATA.DATA_PATH = args.data_path
    if _check_args('zip'):
        config.DATA.ZIP_MODE = True
    if _check_args('cache_mode'):
        config.DATA.CACHE_MODE = args.cache_mode
    if _check_args('pretrained'):
        config.MODEL.PRETRAINED = args.pretrained
    if _check_args('resume'):
        config.MODEL.RESUME = args.resume
    if _check_args('accumulation_steps'):
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if _check_args('use_checkpoint'):
        config.TRAIN.USE_CHECKPOINT = True
    if _check_args('amp_opt_level'):
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if _check_args('disable_amp'):
        config.AMP_ENABLE = False
    if _check_args('output'):
        config.OUTPUT = args.output
    if _check_args('tag'):
        config.TAG = args.tag
    if _check_args('eval'):
        config.EVAL_MODE = True
    if _check_args('throughput'):
        config.THROUGHPUT_MODE = True

    # [SimMIM]
    if _check_args('enable_amp'):
        config.ENABLE_AMP = args.enable_amp

    # for acceleration
    if _check_args('fused_window_process'):
        config.FUSED_WINDOW_PROCESS = True
    if _check_args('fused_layernorm'):
        config.FUSED_LAYERNORM = True
    ## Overwrite optimizer if not None, currently we use it for [fused_adam, fused_lamb]
    if _check_args('optim'):
        config.TRAIN.OPTIMIZER.NAME = args.optim

    # set local rank for distributed training
    if PYTORCH_MAJOR_VERSION == 1:
        config.LOCAL_RANK = args.local_rank
    else:
        config.LOCAL_RANK = int(os.environ['LOCAL_RANK'])

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
