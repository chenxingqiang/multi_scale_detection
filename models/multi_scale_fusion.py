# models/multi_scale_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):
    def __init__(self, in_channels, num_scales=3):
        super(MultiScaleFusion, self).__init__()
        self.num_scales = num_scales
        self.scale_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
                for _ in range(num_scales)
            ]
        )
        self.fusion_conv = nn.Conv2d(
            in_channels * num_scales, in_channels, kernel_size=1, stride=1
        )

    def forward(self, x):
        scale_features = []
        for i in range(self.num_scales):
            scaled_x = F.interpolate(
                x, scale_factor=1 / (2**i), mode="bilinear", align_corners=True
            )
            scale_features.append(self.scale_convs[i](scaled_x))

        concatenated = torch.cat(scale_features, dim=1)
        fused_features = self.fusion_conv(concatenated)
        return fused_features
