# train/train.py

import torch
from torch.utils.data import DataLoader
from models.dbnet import DBNet
from models.titdet import TiTDet
from models.multi_scale_fusion import MultiScaleFusion
from loss.multi_scale_loss.py import MultiScaleLoss
from data.data_loader import TextDataset

def train_model():
    # 加载数据集
    train_dataset = TextDataset('/path/to/dataset')
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 初始化模型
    dbnet = DBNet()
    titdet = TiTDet()
    fusion = MultiScaleFusion(in_channels=512)
    model = nn.Sequential(dbnet, titdet, fusion)

    # 初始化损失函数和优化器
    criterion = MultiScaleLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    model.train()
    for epoch in range(num_epochs):
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Epoch [{epoch}/{num_epochs}], Step [{i}], Loss: {loss.item():.4f}')

if __name__ == "__main__":
    train_model()