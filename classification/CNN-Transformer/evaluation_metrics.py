'''
Module used to evaluate the models performance.
'''

from typing import Dict

import torch
from torch.nn import Module
from torchmetrics.classification import BinaryAccuracy  # same as specificity
from torchmetrics.classification import (BinaryF1Score, BinaryPrecision,
                                         BinaryRecall, BinarySpecificity,
                                         ConfusionMatrix)


class ClassificationMetrics(Module):
    def __init__(self) -> None:
        super().__init__()

        self.acc = BinaryAccuracy()
        self.spec = BinarySpecificity()
        self.precision = BinaryPrecision()
        self.f1 = BinaryF1Score()
        self.recall = BinaryRecall()
        self.conf_matrix = ConfusionMatrix(task="binary", num_classes=2)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, float]:
        prediction = y_true.argmax(dim=1)
        target = y_pred.argmax(dim=1)

        tn, fp, fn, tp = self.conf_matrix(prediction, target).ravel()

        return {"Accuracy": self.acc(prediction, target).item(),
                "Specificity": self.spec(prediction, target).item(),
                "Precision": self.precision(prediction, target).item(),
                "F1": self.f1(prediction, target).item(),
                "Recall": self.recall(prediction, target).item(),
                "ConfusionMatrix": [tn, fp, fn, tp],
                "NPV": (tn/(tn + fn)).tolist()}
