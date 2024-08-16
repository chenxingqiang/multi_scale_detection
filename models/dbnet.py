import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class DBNet(nn.Module):
    def __init__(self, pretrained=True):
        super(DBNet, self).__init__()
        # 使用ResNet34作为骨干网络
        self.backbone = resnet34(pretrained=pretrained)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # 提取骨干网络的特征
        features = self.backbone(x)
        x = F.relu(self.conv1(features))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))  # 使用Sigmoid输出概率图

        return x


# 示例使用
if __name__ == "__main__":
    model = DBNet(pretrained=False)
    dummy_input = torch.randn(1, 3, 640, 640)
    output = model(dummy_input)
    print(output.size())  # 输出应为 (1, 1, 640, 640)
