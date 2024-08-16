import matplotlib.pyplot as plt
import numpy as np


def visualize_results(images, predictions, labels):
    """
    可视化验证结果，包括输入图像、预测结果和实际标签。

    Args:
        images (list): 输入图像。
        predictions (list): 模型的预测结果。
        labels (list): 实际标签。
    """
    batch_size = len(images)
    for i in range(batch_size):
        image = images[i].transpose(1, 2, 0)  # 从CHW变为HWC
        prediction = predictions[i]
        label = labels[i]

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Input Image")

        plt.subplot(1, 3, 2)
        plt.imshow(prediction, cmap="gray")
        plt.title("Prediction")

        plt.subplot(1, 3, 3)
        plt.imshow(label, cmap="gray")
        plt.title("Ground Truth")

        plt.show()
