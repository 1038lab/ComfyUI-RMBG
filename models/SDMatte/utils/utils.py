import os
import json
import torch.nn as nn
from torch.nn import Conv2d
from torch.nn.parameter import Parameter
from diffusers.models.attention_processor import Attention, AttnProcessor
from .replace import custom_prepare_attention_mask, custom_get_attention_scores
import cv2
import torch
import numpy as np


def replace_unet_conv_in(unet, num):
    # replace the first layer to accept 8 in_channels
    _weight = unet.conv_in.weight.clone()  # [320, 4, 3, 3]
    _bias = unet.conv_in.bias.clone()  # [320]
    _weight = _weight.repeat((1, num, 1, 1))  # Keep selected channel(s)
    # half the activation magnitude
    _weight = _weight / num
    # new conv_in channel
    _n_convin_out_channel = unet.conv_in.out_channels
    _new_conv_in = Conv2d(4 * num, _n_convin_out_channel, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    _new_conv_in.weight = Parameter(_weight)
    _new_conv_in.bias = Parameter(_bias)
    unet.conv_in = _new_conv_in
    print("Unet conv_in layer is replaced")
    # replace config
    unet.config["in_channels"] = 4 * num
    print("Unet config is updated")
    return unet


def add_aux_conv_in(unet):
    aux_conv_in = nn.Conv2d(in_channels=4, out_channels=1024, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    aux_conv_in.weight.data[:320, :, :, :] = unet.conv_in.weight.data.clone()
    aux_conv_in.weight.data[320:, :, :, :] = 0.0
    aux_conv_in.bias.data[:320] = unet.conv_in.bias.data.clone()
    aux_conv_in.bias.data[320:] = 0.0
    unet.aux_conv_in = aux_conv_in
    print("add aux_conv_in layer for unet")
    return unet


def replace_attention_mask_method(module, residual_connection):
    if isinstance(module, Attention):
        module.processor = AttnProcessor()
        if hasattr(module, "prepare_attention_mask"):
            module.prepare_attention_mask = custom_prepare_attention_mask.__get__(module)
        if hasattr(module, "cross_attention_dim") and module.cross_attention_dim == 320:
            module.residual_connection = residual_connection
        if hasattr(module, "get_attention_scores"):
            module.get_attention_scores = custom_get_attention_scores.__get__(module)

    # 递归遍历所有子模块
    for child_name, child_module in module.named_children():
        replace_attention_mask_method(child_module, residual_connection)


erosion_kernels = [None] + [cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)) for size in range(1, 30)]


def get_unknown_tensor_from_pred(pred, rand_width=30, train_mode=True):
    ### pred: N, 1 ,H, W
    N, C, H, W = pred.shape

    pred = pred.data.cpu().numpy()
    uncertain_area = np.ones_like(pred, dtype=np.uint8)
    uncertain_area[pred < 1.0 / 255.0] = 0
    uncertain_area[pred > 1 - 1.0 / 255.0] = 0

    for n in range(N):
        uncertain_area_ = uncertain_area[n, 0, :, :]  # H, W
        if train_mode:
            width = np.random.randint(1, rand_width)
        else:
            width = rand_width // 2
        uncertain_area_ = cv2.dilate(uncertain_area_, erosion_kernels[width])
        uncertain_area[n, 0, :, :] = uncertain_area_

    weight = np.zeros_like(uncertain_area)
    weight[uncertain_area == 1] = 1
    weight = torch.from_numpy(weight).float().cuda()
    return weight
