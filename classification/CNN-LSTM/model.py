"""
CNN-LSTM model implemented using PyTorch

Model is based on the CNN-Transformer described in the paper 'A lightweight hybrid CNN-LSTM explainable model for ECG-based arrhythmia detections' by Negin Alamatsaz et. al. (2024)
"""

import torch
from torch.nn import LSTM, Conv1d, Linear, MaxPool1d, Module, ReLU, Sequential


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels, kernel_size) -> None:
        super().__init__()

        self.layer = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
            ),
            ReLU(),
            MaxPool1d(2, 2),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.layer(input)


class CNN(Module):
    def __init__(self) -> None:
        super().__init__()

        self.cnn = Sequential(
            ConvBlock(1, 5, 251),
            ConvBlock(5, 5, 150),
            ConvBlock(5, 10, 100),
            ConvBlock(10, 20, 81),
            ConvBlock(20, 20, 61),
            ConvBlock(20, 10, 14),
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.cnn(input)


class CNNLSTM(Module):
    def __init__(self, lstm_bidirektional=True) -> None:
        super().__init__()

        self.cnn = CNN()

        self.lstmOne = LSTM(
            input_size=10,
            hidden_size=32,
            bidirectional=lstm_bidirektional,
            batch_first=True,
        )
        self.lstmtwo = LSTM(
            input_size=32 * (int(lstm_bidirektional) + 1),
            hidden_size=64,
            bidirectional=lstm_bidirektional,
            batch_first=True,
        )

        self.classifier = Linear(64 * (int(lstm_bidirektional) + 1), 2)

    def encode(self, input) -> torch.Tensor:
        return self.cnn(input)

    def classify(self, input) -> torch.Tensor:
        return self.ffn(input)

    def forward(self, input) -> torch.Tensor:
        features = self.cnn(input)

        hidden1 = None
        hidden2 = None
        for i in range(features.shape[2]):
            lstm_one_output, hidden1 = self.lstmOne(
                torch.unsqueeze(features[:, :, i], dim=1), hidden1
            )
            lstm_two_output, hidden2 = self.lstmtwo(lstm_one_output, hidden2)

        return self.classifier(torch.squeeze(lstm_two_output, dim=1))
