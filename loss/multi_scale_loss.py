# loss/multi_scale_loss.py

import torch
import torch.nn as nn

class MultiScaleLoss(nn.Module):
    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        loss = 0.0
        for pred in predictions:
            loss += self.criterion(pred, targets)
        return loss / len(predictions)