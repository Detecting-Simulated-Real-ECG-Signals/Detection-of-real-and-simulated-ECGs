'''
Custom loss functions developed to improve forecasting performance.
'''

import mlflow
import numpy as np
import torch


class DynamicTargetLoss(torch.nn.Module):
    def __init__(self, target_loss):
        super().__init__()
        self.target_loss = target_loss
        self.last_amount = 1

    def forward(self, prediction, target, current_step):
        loss = None
        for i in range(prediction.shape[1] - self.last_amount):
            loss = torch.mean(
                (
                    prediction[:, : i + self.last_amount]
                    - target[:, : i + self.last_amount]
                )
                ** 2
            )
            if loss.item() > self.target_loss:
                mlflow.log_metric(
                    "prediction_size", self.last_amount, step=current_step
                )
                self.last_amount = self.last_amount + i
                return loss

        mlflow.log_metric("prediction_size",
                          self.last_amount, step=current_step)
        self.last_amount = prediction.shape[1]
        return loss


class WeightedMSELoss(torch.nn.Module):
    def __init__(self, offset, num_predictions, stretch):
        super().__init__()

        log = torch.from_numpy(
            -np.log(np.linspace(1, num_predictions * stretch, num_predictions))
        )
        pos = log - log.min()
        x = (pos / pos.max()) * offset

        self.weights = x / x.sum()

    def forward(self, prediction, target):
        return torch.mean(self.weights * (prediction - target) ** 2)
