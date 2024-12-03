# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform

from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET
from .samplers import SubsetRandomSampler

try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'imagenet22K':
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform)
        nb_classes = 21841
    elif config.DATA.DATASET == 'asianface':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 3
    elif config.DATA.DATASET == 'hjbface':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 2
    elif config.DATA.DATASET == 'agri32k':
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 123
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))

    if config.DATA.NOISE_MODEL != 'norm':
        if config.DATA.NOISE_MODEL == 'salt_pepper':
            t.append(transforms.Lambda(lambda img: add_salt_pepper_noise(image=img, amount=config.DATA.NOICE_AMOUNT)))
        elif config.DATA.NOISE_MODEL == 'gaussian':
            t.append(transforms.Lambda(lambda img: add_gaussian_noise(image=img, mean=config.DATA.NOICE_MEAN, std=config.DATA.NOICE_STD)))
        elif config.DATA.NOISE_MODEL == 'poisson':
            t.append(transforms.Lambda(
                lambda img: add_poisson_noise(image=img)))
        elif config.DATA.NOISE_MODEL == 'laplacian':
            t.append(transforms.Lambda(lambda img: add_laplacian_noise(image=img, loc=config.DATA.NOICE_LOC, scale=config.DATA.NOICE_SCALE)))
        elif config.DATA.NOISE_MODEL == 'speckle':
            t.append(transforms.Lambda(
                lambda img: add_speckle_noise(image=img, mean=config.DATA.NOICE_MEAN, std=config.DATA.NOICE_STD)))
        elif config.DATA.NOISE_MODEL == 'mixed':
            t.append(transforms.Lambda(
                lambda img: add_mixed_noise(image=img, noise_types=config.DATA.NOICE_TYPES)))
        else:
            pass
    return transforms.Compose(t)

def add_gaussian_noise(image, mean=0., std=0.1):
    noise = torch.randn_like(image) * std + mean
    noisy_image = image + noise
    return noisy_image


def scale_to_0_1(image):
    return (image + 1) / 2


def scale_to_neg1_pos1(image):
    return image * 2 - 1


def add_salt_pepper_noise(image, amount=0.05):
    image = scale_to_0_1(image)

    # 获取图像的尺寸
    C, H, W = image.shape

    # 为每个像素点生成一个随机数
    noise = torch.rand_like(image)

    # 确定哪些像素点将变为椒噪声
    pepper_mask = noise < (amount / 2)
    # 确定哪些像素点将变为盐噪声
    salt_mask = noise > (1 - amount / 2)

    # 将椒噪声位置的像素值设为 0
    image[pepper_mask] = 0
    # 将盐噪声位置的像素值设为 1
    image[salt_mask] = 1

    return scale_to_neg1_pos1(image)


def add_laplacian_noise(image, loc=0., scale=1.):
    laplace_dist = torch.distributions.laplace.Laplace(loc, scale)
    noise = laplace_dist.sample(image.size())
    noisy_image = image + noise
    return noisy_image

def add_poisson_noise(image):
    image_np = image.cpu().numpy()  # 确保 image 在 CPU
    counts = np.random.poisson(image_np * 255) / 255.0  # 假设图像值在 0 到 1 之间
    noisy_image = torch.from_numpy(counts).to(image.device)
    # 确保 noisy_image 的类型与 original image 相同
    noisy_image = noisy_image.type_as(image)
    return noisy_image

def add_speckle_noise(image, mean=0., std=0.1):
    noise = torch.randn(*image.shape) * std + mean
    noisy_image = image + image * noise
    return noisy_image

def add_mixed_noise(image, noise_types=['gaussian', 'poisson']):
    noisy_image = image
    for noise_type in noise_types:
        if noise_type == 'gaussian':
            noisy_image = add_gaussian_noise(noisy_image)
        elif noise_type == 'salt_pepper':
            noisy_image = add_salt_pepper_noise(noisy_image)
        elif noise_type == 'poisson':
            noisy_image = add_poisson_noise(noisy_image)
        elif noise_type == 'laplacian':
            noisy_image = add_laplacian_noise(noisy_image)
        elif noise_type == 'speckle':
            noisy_image = add_speckle_noise(noisy_image)
        elif noise_type == 'uniform':
            noisy_image = add_uniform_noise(noisy_image)
        elif noise_type == 'quantization':
            noisy_image = add_quantization_noise(noisy_image)
        else:
            pass
    return noisy_image

def add_uniform_noise(image, a=-0.1, b=0.1):
    uniform_dist = torch.distributions.uniform.Uniform(a, b)
    noise = uniform_dist.sample(image.size())
    noisy_image = image + noise
    return noisy_image

def add_quantization_noise(image, bits=8):
    quantized_image = torch.round(image * (2**bits - 1)) / (2**bits - 1)
    noisy_image = quantized_image
    return noisy_image

