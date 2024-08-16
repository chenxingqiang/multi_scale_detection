import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class TiTDet(nn.Module):
    def __init__(self, pretrained=True):
        super(TiTDet, self).__init__()
        # 使用MobileNetV2作为骨干网络
        self.backbone = mobilenet_v2(pretrained=pretrained).features
        self.conv1 = nn.Conv2d(1280, 512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 1, kernel_size=1)

        # 差异化二值化模块 (Th-DB)
        self.thresh_conv1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.thresh_conv2 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        score_map = torch.sigmoid(self.conv4(x))  # 输出得分图

        # 差异化二值化
        thresh = F.relu(self.thresh_conv1(x))
        thresh = torch.sigmoid(self.thresh_conv2(thresh))

        return score_map, thresh


# 示例使用
if __name__ == "__main__":
    model = TiTDet(pretrained=False)
    dummy_input = torch.randn(1, 3, 640, 640)
    score_map, thresh = model(dummy_input)
    print(
        score_map.size(), thresh.size()
    )  # 输出应为 (1, 1, 640, 640) 和 (1, 1, 640, 640)
