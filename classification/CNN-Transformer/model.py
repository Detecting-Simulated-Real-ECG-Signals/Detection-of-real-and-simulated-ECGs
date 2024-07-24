"""
CNN-Transformer model implemented using PyTorch

Model is based on the CNN-Transformer described in the paper 'A transformer-based deep neural network for arrhythmia detection using continuous ECG signals' by Rui Hu et. al. (2022)
"""

import math
from typing import List

import torch
import torch.nn.functional as F
from torch.nn import (BatchNorm1d, Conv1d, Dropout, Linear, Module, ReLU,
                      Sequential, Sigmoid, Transformer)


class SpatialSELayer1d(
    Module
):  # by https://github.com/ioanvl/1d_channel-spatial_se_layers/blob/master/se_layers/se_layers.py

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer1d, self).__init__()
        self.conv = Conv1d(num_channels, 1, 1)
        self.sigmoid = Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        return input_tensor * squeeze_tensor.view(batch_size, 1, a)


class Bottleneck(Module):
    def __init__(self, in_channels, out_channels, stride, kernal_size, k) -> None:
        super().__init__()

        self.in_channels = in_channels

        self.conv1 = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=1,
            ),
            BatchNorm1d(out_channels),
            ReLU(),
        )
        self.depthwise_conv = Sequential(
            Conv1d(
                in_channels=out_channels,
                out_channels=out_channels * k,
                stride=1,
                kernel_size=kernal_size,
                groups=out_channels,
                padding=int(kernal_size / 2),
            ),
            BatchNorm1d(out_channels * k),
            ReLU(),
        )
        self.conv2 = Sequential(
            Conv1d(
                in_channels=out_channels * k,
                out_channels=out_channels,
                stride=1,
                kernel_size=1,
            ),
            BatchNorm1d(out_channels),
            ReLU(),
        )

        self.se = SpatialSELayer1d(out_channels)

        # downsample
        self.res_connection = Sequential(  # https://arxiv.org/pdf/1512.03385#page=9&zoom=100,412,532 use projection shortcuts because it is a treadoff between computational cost and performance
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=1,
            ),
            BatchNorm1d(out_channels),
        )

    def forward(self, input) -> torch.Tensor:

        residual = input.clone()

        x1 = self.conv1(input)
        x2 = self.depthwise_conv(x1)
        x3 = self.conv2(x2)

        se_r = self.se(x3)

        r = self.res_connection(residual)
        return se_r + r  # different resolution, not possible to add!


class bottleneckCNN(Module):
    def __init__(
        self, d_model, width_multiplier=1, stride=2, kernel_size=3, k=6, dropout=0.1
    ) -> None:
        super().__init__()

        self.bottlenecks = Sequential(
            Conv1d(
                in_channels=1,
                out_channels=int(8 * width_multiplier),
                stride=2,
                kernel_size=1,
            ),
            BatchNorm1d(int(8 * width_multiplier)),
            ReLU(),
            Bottleneck(
                int(8 * width_multiplier),
                int(8 * width_multiplier),
                stride,
                kernel_size,
                k,
            ),
            Bottleneck(
                int(8 * width_multiplier),
                int(16 * width_multiplier),
                stride,
                kernel_size,
                k,
            ),
            Bottleneck(
                int(16 * width_multiplier),
                int(32 * width_multiplier),
                stride,
                kernel_size,
                k,
            ),
            Bottleneck(
                int(32 * width_multiplier),
                int(64 * width_multiplier),
                stride,
                kernel_size,
                k,
            ),
            Bottleneck(
                int(64 * width_multiplier),
                int(128 * width_multiplier),
                stride,
                kernel_size,
                k,
            ),
            Bottleneck(
                int(128 * width_multiplier),
                int(256 * width_multiplier),
                1,
                kernel_size,
                k,
            ),
            Conv1d(
                in_channels=int(256 * width_multiplier),
                out_channels=d_model,
                stride=1,
                kernel_size=1,
            ),
            BatchNorm1d(d_model),
        )

    def forward(self, input) -> torch.Tensor:
        return self.bottlenecks(input)


class PositionalEncoding(
    Module
):  # from https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class reshapeLayers(Module):
    """
    switch inputs dimentions 2 and 3\n
    input: N, C, T\n
    output: N, T, C\n
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input):
        return torch.reshape(input, (input.shape[0], input.shape[2], input.shape[1]))


class CNNTransformer(Module):
    def __init__(
        self,
        d_model: int = 512,
        stride: int = 2,
        kernel_size: int = 3,
        nhead: int = 8,
        num_layers: int = 6,
        width_multiplier: int = 1,
        dropout: float = 0.1,
        FFN_dim_hidden_layers: List[int] = [],
    ) -> None:
        super().__init__()

        self.cnn = bottleneckCNN(
            d_model, width_multiplier, stride, kernel_size, dropout=dropout
        )

        self.reshape = reshapeLayers()

        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.encoder = Transformer(
            d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=0,
            batch_first=True,
        ).encoder

        # detemine linear layer size by testing and build FFN acording to 'FFN_dim_hidden_layers'
        FFN_dim_hidden_layers.append(2)
        linear_layers = [Linear(d_model, FFN_dim_hidden_layers[0])]

        for i, d in enumerate(FFN_dim_hidden_layers[:-1]):
            linear_layers.append(ReLU())
            linear_layers.append(Linear(d, FFN_dim_hidden_layers[i + 1]))

        self.ffn = Sequential(*linear_layers)

    def forward(self, input) -> torch.Tensor:
        features = self.cnn(input)

        # reshape for encoder
        r_features = torch.reshape(
            features, (features.shape[0], features.shape[2], features.shape[1])
        )

        p_features = self.pos_encoder(r_features)

        e_data = self.encoder(p_features)

        return self.ffn(torch.mean(e_data, dim=1))
