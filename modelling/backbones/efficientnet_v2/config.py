import os
import re
import subprocess
from pathlib import Path
import numpy as np
import torch

def get_efficientnet_v2_hyperparam(model_name):
    from box import Box
    # train_size, eval_size, dropout, randaug, mixup
    if 'efficientnet_v2_s' in model_name:
        end = 300, 384, 0.2, 10, 0
    elif 'efficientnet_v2_m' in model_name:
        end = 384, 480, 0.3, 15, 0.2
    elif 'efficientnet_v2_l' in model_name:
        end = 384, 480, 0.4, 20, 0.5
    elif 'efficientnet_v2_xl' in model_name:
        end = 384, 512, 0.4, 20, 0.5
    return Box({"init_train_size": 128, "init_dropout": 0.1, "init_randaug": 5, "init_mixup": 0,
             "end_train_size": end[0], "end_dropout": end[2], "end_randaug": end[3], "end_mixup": end[4], "eval_size": end[1]})


def get_efficientnet_v2_structure(model_name):
    if 'efficientnet_v2_s' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 2, False, True),
            (4, 3, 2, 24, 48, 4, False, True),
            (4, 3, 2, 48, 64, 4, False, True),
            (4, 3, 2, 64, 128, 6, True, False),
            (6, 3, 1, 128, 160, 9, True, False),
            (6, 3, 2, 160, 256, 15, True, False),
        ]
    elif 'efficientnet_v2_m' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 24, 24, 3, False, True),
            (4, 3, 2, 24, 48, 5, False, True),
            (4, 3, 2, 48, 80, 5, False, True),
            (4, 3, 2, 80, 160, 7, True, False),
            (6, 3, 1, 160, 176, 14, True, False),
            (6, 3, 2, 176, 304, 18, True, False),
            (6, 3, 1, 304, 512, 5, True, False),
        ]
    elif 'efficientnet_v2_l' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 7, False, True),
            (4, 3, 2, 64, 96, 7, False, True),
            (4, 3, 2, 96, 192, 10, True, False),
            (6, 3, 1, 192, 224, 19, True, False),
            (6, 3, 2, 224, 384, 25, True, False),
            (6, 3, 1, 384, 640, 7, True, False),
        ]
    elif 'efficientnet_v2_xl' in model_name:
        return [
            # e k  s  in  out xN  se   fused
            (1, 3, 1, 32, 32, 4, False, True),
            (4, 3, 2, 32, 64, 8, False, True),
            (4, 3, 2, 64, 96, 8, False, True),
            (4, 3, 2, 96, 192, 16, True, False),
            (6, 3, 1, 192, 256, 24, True, False),
            (6, 3, 2, 256, 512, 32, True, False),
            (6, 3, 1, 512, 640, 8, True, False),
        ]
    
model_urls = {
    "efficientnet_v2_s": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-s.npy",
    "efficientnet_v2_m": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-m.npy",
    "efficientnet_v2_l": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-l.npy",
    "efficientnet_v2_s_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-s-21k.npy",
    "efficientnet_v2_m_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-m-21k.npy",
    "efficientnet_v2_l_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-l-21k.npy",
    "efficientnet_v2_xl_in21k": "https://github.com/hankyul2/EfficientNetV2-pytorch/releases/download/EfficientNetV2-pytorch/efficientnetv2-xl-21k.npy",
}


def load_from_zoo(model, model_name, pretrained_path='pretrained/official'):
    Path(os.path.join(pretrained_path, model_name)).mkdir(parents=True, exist_ok=True)
    file_name = os.path.join(pretrained_path, model_name, os.path.basename(model_urls[model_name]))
    load_npy(model, load_npy_from_url(url=model_urls[model_name], file_name=file_name))


def load_npy_from_url(url, file_name):
    if not Path(file_name).exists():
        subprocess.run(["wget", "-r", "-nc", '-O', file_name, url])
    return np.load(file_name, allow_pickle=True).item()


def npz_dim_convertor(name, weight):
    weight = torch.from_numpy(weight)
    if 'kernel' in name:
        if weight.dim() == 4:
            if weight.shape[3] == 1:
                # depth-wise convolution 'h w in_c out_c -> in_c out_c h w'
                # weight = torch.permute(weight, (2, 3, 0, 1))
                weight = weight.permute((2, 3, 0, 1))
            else:
                # 'h w in_c out_c -> out_c in_c h w'
                # weight = torch.permute(weight, (3, 2, 0, 1))
                weight = weight.permute((3, 2, 0, 1))
        elif weight.dim() == 2:
            weight = weight.transpose(1, 0)
    elif 'scale' in name or 'bias' in name:
        weight = weight.squeeze()
    return weight


def load_npy(model, weight):
    name_convertor = [
        # stem
        ('stem.0.weight', 'stem/conv2d/kernel/ExponentialMovingAverage'),
        ('stem.1.weight', 'stem/tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('stem.1.bias', 'stem/tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('stem.1.running_mean', 'stem/tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('stem.1.running_var', 'stem/tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # fused layer
        ('block.fused.0.weight', 'conv2d/kernel/ExponentialMovingAverage'),
        ('block.fused.1.weight', 'tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('block.fused.1.bias', 'tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('block.fused.1.running_mean', 'tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('block.fused.1.running_var', 'tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # linear bottleneck
        ('block.linear_bottleneck.0.weight', 'conv2d/kernel/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.weight', 'tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.bias', 'tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.running_mean', 'tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('block.linear_bottleneck.1.running_var', 'tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # depth wise layer
        ('block.depth_wise.0.weight', 'depthwise_conv2d/depthwise_kernel/ExponentialMovingAverage'),
        ('block.depth_wise.1.weight', 'tpu_batch_normalization_1/gamma/ExponentialMovingAverage'),
        ('block.depth_wise.1.bias', 'tpu_batch_normalization_1/beta/ExponentialMovingAverage'),
        ('block.depth_wise.1.running_mean', 'tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage'),
        ('block.depth_wise.1.running_var', 'tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage'),

        # se layer
        ('block.se.fc1.weight', 'se/conv2d/kernel/ExponentialMovingAverage'), ('block.se.fc1.bias', 'se/conv2d/bias/ExponentialMovingAverage'),
        ('block.se.fc2.weight', 'se/conv2d_1/kernel/ExponentialMovingAverage'), ('block.se.fc2.bias', 'se/conv2d_1/bias/ExponentialMovingAverage'),

        # point wise layer
        ('block.fused_point_wise.0.weight', 'conv2d_1/kernel/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.weight', 'tpu_batch_normalization_1/gamma/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.bias', 'tpu_batch_normalization_1/beta/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.running_mean', 'tpu_batch_normalization_1/moving_mean/ExponentialMovingAverage'),
        ('block.fused_point_wise.1.running_var', 'tpu_batch_normalization_1/moving_variance/ExponentialMovingAverage'),

        ('block.point_wise.0.weight', 'conv2d_1/kernel/ExponentialMovingAverage'),
        ('block.point_wise.1.weight', 'tpu_batch_normalization_2/gamma/ExponentialMovingAverage'),
        ('block.point_wise.1.bias', 'tpu_batch_normalization_2/beta/ExponentialMovingAverage'),
        ('block.point_wise.1.running_mean', 'tpu_batch_normalization_2/moving_mean/ExponentialMovingAverage'),
        ('block.point_wise.1.running_var', 'tpu_batch_normalization_2/moving_variance/ExponentialMovingAverage'),

        # head
        ('head.bottleneck.0.weight', 'head/conv2d/kernel/ExponentialMovingAverage'),
        ('head.bottleneck.1.weight', 'head/tpu_batch_normalization/gamma/ExponentialMovingAverage'),
        ('head.bottleneck.1.bias', 'head/tpu_batch_normalization/beta/ExponentialMovingAverage'),
        ('head.bottleneck.1.running_mean', 'head/tpu_batch_normalization/moving_mean/ExponentialMovingAverage'),
        ('head.bottleneck.1.running_var', 'head/tpu_batch_normalization/moving_variance/ExponentialMovingAverage'),

        # classifier
        ('head.classifier.weight', 'head/dense/kernel/ExponentialMovingAverage'),
        ('head.classifier.bias', 'head/dense/bias/ExponentialMovingAverage'),

        ('\\.(\\d+)\\.', lambda x: f'_{int(x.group(1))}/'),
    ]

    for name, param in list(model.named_parameters()) + list(model.named_buffers()):
        for pattern, sub in name_convertor:
            name = re.sub(pattern, sub, name)
        if 'dense/kernel' in name and list(param.shape) not in [[1000, 1280], [21843, 1280]]:
            continue
        if 'dense/bias' in name and list(param.shape) not in [[1000], [21843]]:
            continue
        if 'num_batches_tracked' in name:
            continue
        param.data.copy_(npz_dim_convertor(name, weight.get(name)))