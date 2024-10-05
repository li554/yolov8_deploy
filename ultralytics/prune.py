import os

import torch
from tqdm import tqdm

from relation import match_bboxes, find_parent_nodes, visualize_nodes, metric
from ultralytics import YOLO
from ultralytics.nn.modules import Bottleneck, Conv, C2f, SPPF, Detect


def prune_conv(conv1: Conv, conv2: Conv, threshold=0.01):
    # 剪枝top-bottom conv结构
    # 首先，剪枝conv1的bn层和conv层
    # 获取conv1的bn层权重和偏置参数作为剪枝的依据
    gamma = conv1.bn.weight.data.detach()
    beta = conv1.bn.bias.data.detach()
    # 索引列表，用于存储剪枝后保留的参数索引
    keep_idxs = []
    local_threshold = threshold
    # 保证剪枝后的通道数不少于8，便于硬件加速
    while len(keep_idxs) < 8:
        # 取绝对值大于阈值的参数对应的索引
        keep_idxs = torch.where(gamma.abs() >= local_threshold)[0]
        # 降低阈值
        local_threshold = local_threshold * 0.5
    # print(local_threshold)
    # 剪枝后的通道数
    n = len(keep_idxs)
    # 更新BN层参数
    conv1.bn.weight.data = gamma[keep_idxs]
    conv1.bn.bias.data = beta[keep_idxs]
    conv1.bn.running_var.data = conv1.bn.running_var.data[keep_idxs]
    conv1.bn.running_mean.data = conv1.bn.running_mean.data[keep_idxs]
    conv1.bn.num_features = n
    # 更新conv层权重
    conv1.conv.weight.data = conv1.conv.weight.data[keep_idxs]
    # 更新conv层输出通道数
    conv1.conv.out_channels = n
    # 更新conv层偏置，如果存在的话
    if conv1.conv.bias is not None:
        conv1.conv.bias.data = conv1.conv.bias.data[keep_idxs]

    # 然后，剪枝conv2的conv层
    if not isinstance(conv2, list):
        conv2 = [conv2]

    for item in conv2:
        if item is not None:
            if isinstance(item, Conv):
                conv = item.conv
            else:
                conv = item
            # 更新输入通道数
            conv.in_channels = n
            # 更新权重
            conv.weight.data = conv.weight.data[:, keep_idxs]


def prune_module(m1, m2, threshold=0.01):
    # 剪枝 模块间衔接处结构，m1需要获取模块的bottom conv，m2需要获取模块的top conv
    # 根据yolov8的结构，可能存在的组合有
    # Conv-Conv, Conv-C2f, C2f-Conv, C2f-SPPF, SPPF-Conv, SPPF-C2f
    # 打印出m1和m2的名字
    print(m1.__class__.__name__, end="->")
    if isinstance(m2, list):
        print([item.__class__.__name__ for item in m2])
    else:
        print(m2.__class__.__name__)
    if isinstance(m1, C2f):  # C2f as a top conv
        m1 = m1.cv2
    if not isinstance(m2, list):  # m2 is just one module
        m2 = [m2]
    for i, item in enumerate(m2):
        if isinstance(item, C2f) or isinstance(item, SPPF):
            m2[i] = item.cv1
    prune_conv(m1, m2, threshold)


def prune_model(yolo, save_dir, factor=0.8):
    model = yolo.model

    # 统计所有的BN层权重和偏置参数
    ws = []
    bs = []

    for name, m in model.named_modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            w = m.weight.abs().detach()
            b = m.bias.abs().detach()
            ws.append(w)
            bs.append(b)
            # print(name, w.max().item(), w.min().item(), b.max().item(), b.min().item())

    ws = torch.cat(ws)
    # 从大到小排序，取80%的参数对应的阈值
    threshold = torch.sort(ws, descending=True)[0][int(len(ws) * factor)]
    print(threshold)

    # 先剪枝整个网络bottleneck模块内部的结构
    for name, m in model.named_modules():
        if isinstance(m, Bottleneck):
            prune_conv(m.cv1, m.cv2, threshold)

    # 再剪枝backbone模块间衔接结构
    seq = model.model
    for i in range(3, 9):
        if i in [6, 4, 9]: continue
        prune_module(seq[i], seq[i + 1], threshold)

    # 再剪枝Head模块间衔接结构
    # Head模块间剪枝包括两部分，一部分是相邻下层连接，一部分是跨层到Detect层的输出
    # 从last_inputs到colasts是相邻下层连接，从last_inputs到detect是跨层到最后的输出
    detect: Detect = seq[-1]
    last_inputs = [seq[15], seq[18], seq[21]]
    colasts = [seq[16], seq[19], None]
    for last_input, colast, cv2, cv3 in zip(last_inputs, colasts, detect.cv2, detect.cv3):
        prune_module(last_input, [colast, cv2[0], cv3[0]], threshold)
        # 剪枝Detect层内部模块间衔接结构
        prune_module(cv2[0], cv2[1], threshold, )
        prune_module(cv2[1], cv2[2], threshold)
        prune_module(cv3[0], cv3[1], threshold)
        prune_module(cv3[1], cv3[2], threshold)

    # 设置所有参数为可训练，为retrain做准备
    for name, p in model.named_parameters():
        p.requires_grad = True

    if save_dir:
        # 保存剪枝后的模型
        yolo.save(save_dir)
