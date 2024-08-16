import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class TextDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集的根目录。
            transform (callable, optional): 应用于样本的变换（数据增强）。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        """
        加载数据集中的图像路径和标签。
        """
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith((".jpg", ".png", ".jpeg")):
                    image_path = os.path.join(subdir, file)
                    label_path = (
                        image_path.replace(".jpg", ".txt")
                        .replace(".png", ".txt")
                        .replace(".jpeg", ".txt")
                    )

                    if os.path.exists(label_path):
                        self.image_paths.append(image_path)
                        self.labels.append(self._load_label(label_path))

    def _load_label(self, label_path):
        """
        加载标签文件。
        Args:
            label_path (string): 标签文件的路径。

        Returns:
            list: 标签数据，通常是边界框的坐标。
        """
        with open(label_path, "r") as f:
            labels = []
            for line in f:
                coords = list(map(float, line.strip().split(",")))
                labels.append(coords)
        return labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        获取一个样本。
        Args:
            idx (int): 样本的索引。

        Returns:
            tuple: (图像, 标签)
        """
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = torch.tensor(self.labels[idx], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, label


# 示例使用：
if __name__ == "__main__":
    transform = transforms.Compose(
        [
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
        ]
    )

    dataset = TextDataset(root_dir="/path/to/dataset", transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    for images, labels in dataloader:
        print(images.size(), labels.size())
