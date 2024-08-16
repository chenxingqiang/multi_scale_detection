import torch
from torch.utils.data import DataLoader
from models.dbnet import DBNet
from models.titdet import TiTDet
from models.multi_scale_fusion import MultiScaleFusion
from data.data_loader import TextDataset
from utils.metrics import calculate_metrics


def validate_model():
    # 加载数据集
    val_dataset = TextDataset("/path/to/validation/dataset")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    # 初始化模型
    dbnet = DBNet(pretrained=False)
    titdet = TiTDet(pretrained=False)
    fusion = MultiScaleFusion(in_channels=512)
    model = nn.Sequential(dbnet, titdet, fusion)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load("/path/to/saved/model.pth"))
    model.eval()

    all_predictions = []
    all_labels = []

    # 验证循环
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            all_predictions.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 计算评估指标
    metrics = calculate_metrics(all_predictions, all_labels)
    print(f"Validation Metrics: {metrics}")


if __name__ == "__main__":
    validate_model()
