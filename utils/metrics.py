import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def calculate_metrics(predictions, labels):
    """
    计算验证过程中的评估指标，包括精确度、召回率和F1分数。

    Args:
        predictions (list): 预测的结果。
        labels (list): 实际标签。

    Returns:
        dict: 各项评估指标。
    """
    # 转换为二值形式，假设阈值为0.5
    predictions = (np.array(predictions) > 0.5).astype(int)
    labels = np.array(labels).astype(int)

    precision = precision_score(labels.flatten(), predictions.flatten())
    recall = recall_score(labels.flatten(), predictions.flatten())
    f1 = f1_score(labels.flatten(), predictions.flatten())

    return {"precision": precision, "recall": recall, "f1_score": f1}
