"""
Multiple model architectures trained and evaluated during the generative approach. All models are implemented using PyTorch.
"""

from typing import Tuple

import torch
from torch import nn

# support classes


class SwitchDimTwoAndThree(torch.nn.Module):
    def forward(self, x) -> torch.Tensor:
        return torch.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))


# Models


class EncoderDecoderLSTM(nn.Module):
    def __init__(self, lstm_hidden_size, num_lstm_layers, bidirectional) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(lstm_hidden_size * (int(bidirectional) + 1), 1)

    def forward(self, x, num_predictions) -> torch.Tensor:
        output, hidden = self.encode(torch.unsqueeze(x, dim=1))

        predictions, hidden = self.decode(
            torch.unsqueeze(x[:, -1], dim=1), hidden)
        for i in range(num_predictions - 1):
            d_output, hidden = self.decode(
                torch.unsqueeze(predictions[:, -1], dim=1), hidden
            )
            predictions = torch.concat([predictions, d_output], dim=1)

        return predictions

    def decode(self, input_value, hidden) -> Tuple[torch.Tensor, torch.Tensor]:
        d_output, hidden = self.decoder(
            torch.unsqueeze(input_value, dim=1), hidden)
        return self.linear(torch.squeeze(d_output, dim=1)), hidden

    def encode(self, x) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        assert len(x.shape) == 3, "input must be shape 3"

        hidden: Tuple[torch.Tensor, torch.Tensor] = None
        for i in range(x.shape[1]):
            output, hidden = self.encoder(
                torch.unsqueeze(x[:, i], dim=2), hidden)
        return (output, hidden)


class CustomLSTM(nn.Module):
    def __init__(self, lstm_hidden_size, num_lstm_layers, bidirectional, input_size=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(lstm_hidden_size * (int(bidirectional) + 1), 1)

    def forward(self, x, hidden):  # x [N, L]
        lstm_out, hidden = self.lstm(
            torch.unsqueeze(x, dim=1), hidden)  # [N L, 1]
        return self.linear(torch.squeeze(lstm_out, dim=1)), hidden


class CnnLSTM(nn.Module):
    def __init__(
        self, cnn_kernels, lstm_hidden_size, num_lstm_layers, bidirectional
    ) -> None:
        super().__init__()

        self.cnn = torch.nn.Sequential(
            torch.nn.Conv1d(1, cnn_kernels, 1), torch.nn.ReLU(
            ), SwitchDimTwoAndThree()
        )

        self.lstm = CustomLSTM(
            lstm_hidden_size, num_lstm_layers, bidirectional, cnn_kernels
        )

    def forward(self, input, num_predictions) -> torch.Tensor:
        c_output = self.cnn(torch.unsqueeze(input, dim=1))

        # bild memory
        hidden = None
        for i in range(input.shape[1]):
            predictions, hidden = self.lstm(c_output[:, i], hidden)

        # predict
        for i in range(1, num_predictions):
            c_output = self.cnn(
                torch.unsqueeze(torch.unsqueeze(
                    predictions[:, -1], dim=1), dim=1)
            )
            d_output, hidden = self.lstm(
                torch.squeeze(c_output, dim=1), hidden)
            predictions = torch.concat([predictions, d_output], dim=1)

        return predictions


class TCN(nn.Module):
    def __init__(self, conv_channels, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size

        self.c1 = nn.Conv1d(1, conv_channels, kernel_size, dilation=1)
        self.c2 = nn.Conv1d(conv_channels, conv_channels,
                            kernel_size, dilation=2)
        self.c3 = nn.Conv1d(conv_channels, conv_channels,
                            kernel_size, dilation=4)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.c3(self.c2(self.c1(X)))

    def field_of_view(self) -> int:
        return 1 + ((self.kernel_size - 1) * (4 + 2 + 1))


class manual_LSTM(nn.Module):
    def __init__(self, first_layer, second_layer):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=1, hidden_size=first_layer, num_layers=1, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=first_layer,
            hidden_size=second_layer,
            num_layers=1,
            batch_first=True,
        )
        self.linear = nn.Linear(second_layer, 1)

    def forward(
        self, X: torch.Tensor, hidden
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out1, hidden1 = self.lstm1(X, hidden[0])
        out2, hidden2 = self.lstm2(out1, hidden[1])

        out = self.linear(out2)
        return out, (hidden1, hidden2)

    def predict(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        out1, hidden1 = self.lstm1(X)
        out2, hidden2 = self.lstm2(out1)

        out = self.linear(out2)
        return out, (hidden1, hidden2)


class TCN_LSTM(
    nn.Module
):  # paper: https://pdf.sciencedirectassets.com/272229/1-s2.0-S1568494622X00185/1-s2.0-S1568494622009942/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEKH%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFImGqV84%2BunWrr5LRpkO6qabmGwVXRTiUtv2Jscf4xZAiBeOJZpL1pbk4XqiS%2Bx1UORBotRCghVmAnAIABmo3WeRyq8BQi5%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAUaDDA1OTAwMzU0Njg2NSIM3JSDg8IPQ4TZ7C2LKpAFdGOAWXvp%2BnsGwC0%2FgCSLElB2fSMcrAULaMG0IFKEcvVC2Vz5l7%2FNV9CVUkMVNJwRoLoTEayy8Rz0Cdwbqlvrxw3ElvrcNdBi07G3VrnKjl%2FMOohLr52npKisNWItauy6z55DDH1vpkMggkjt%2BDymd%2BGuPvquqeITeO1FPBDijSiLybPj35uF%2B087dOcxqTXEwhTGWw3eF0We8hSWBMIXrG%2FAzpf571H2%2Fmvh47bwVaGmUz2YXsUhcWp2vO5dURIbDah1msM0ji9WJocWUr7LifvlE2BE9v14%2F9F2R5V3FMn8pbzsMenJGIn5FhAyDijHQa8XTnCGcfsIeo2e%2BHbobNFOp9LrANB25Q%2FVGvvLGELvoOkBnG6vbGdHZkcdGvT3N3AcBdqueq5CBNhQOLsq%2BIzB2pFm68mfMLHnPq1PjoIrGAJ5V5mBy0yInm3YK1OXhj50zo8RPY8tqAi6smknHqZbhRYO08v2HxYHwRjK5WCB2DcNZx4IqBG4zchkiRh8CWIkBVjshRaZgf4%2FHTul6RYDGLTO6JU98aIOPdjd7REpMm%2FabfRJ2vYzmEpCDNcD7NADrWWXf4lnarta1ieSrj9eIzMaophEzg4GjQ8zKM3xTlvD4bU2eHyI3aihswAZS%2BwWsWIoJsuD21Fsc9%2BbQ1WT5ToTM10XfV9UFus%2Br44Cqi8hWuhRUvNIQ6Y88fPc7F7nBOjYKGr5DzSpO%2BdeXTntxUprecfYqnd2ujhnL5dLv8tJvw84nkD%2F7mARMGmnL8Ez5x96OBgtHkKTSer0UNrAHgB5dGj1ktQNq2hOVz953jof%2Bxsr%2BCT3ZXFYpuXPiuir5XTxSOlIpQecpgnd9WNjSQCJ4w56KNqimC4mXsAwzqaPsAY6sgHv7m%2F0G9PhQ5udGvMGzM3iw%2FiDKiRwWRWOX%2BC9aqdGbd27AEQjiE7th4rcDvq7KFTOt89V4mfrqznyfPU0er3DJqi78o3eE06Or6S9V%2BmIIJgSX%2FI8ZXcbZsm0s6gFV0%2Bx3sbdw9tOLcJ9m7uu2UOnDGsE%2BykUQR6z%2FElsYbPb9CuEI3Fkj%2FwXwRMuHQhuzfag7Efyujgi6BwBEadCLtD9DlICaKWaLhV53pzavvi%2FKySB&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20240327T081920Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTY6JWWDIMU%2F20240327%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=90995fb086d5f89cfa332c502f0f249762288fcdf20c2727b9350c73e7b621d2&hash=d3674d0589acbbe6f6deade83db945ef3e4c8a0b08044dbf1cc88f01b7d6714e&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S1568494622009942&tid=spdf-a37feec5-dc91-4349-b1fe-c908d543bd8b&sid=8fffb55c17dd304a53497f669321f38a194agxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1e035d500606015a5352&rr=86adf4722af9b3ad&cc=de
    def __init__(
        self,
        conv_channels,
        kernel_size,
        lstm_hidden_size,
        num_lstm_layers,
        bidirectional,
    ):
        super().__init__()

        self.tcn = TCN(conv_channels, kernel_size)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.linear = nn.Linear(lstm_hidden_size * (int(bidirectional) + 1), 1)

    def build_memory(
        self, X: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        c = self.relu(self.tcn(torch.unsqueeze(X, dim=1)))

        # build memory
        out, hidden = self.lstm(torch.unsqueeze(c[:, :, 0], dim=1))
        predictions = self.linear(torch.squeeze(out, dim=1))

        for i in range(1, c.shape[2]):
            out, hidden = self.lstm(torch.unsqueeze(c[:, :, 0], dim=1), hidden)
            predictions = torch.concat(
                [predictions, self.linear(torch.squeeze(out, dim=1))], dim=1
            )

        return predictions, hidden

    def predict(self, start_values, hidden, num_predictions) -> torch.Tensor:
        fov = self.relu(self.tcn.field_of_view())
        assert (
            start_values.shape[1] == fov
        ), f"start_values must be of shape (B, {fov}), got {start_values.shape}"
        start_values = start_values.detach()

        hidden = None
        for i in range(num_predictions):
            c = self.relu(self.tcn(torch.unsqueeze(
                start_values[:, -fov:], dim=1)))

            lstm_out_N, hidden = self.lstm(
                torch.unsqueeze(c[:, :, 0], dim=1), hidden)
            start_values = torch.concat(
                [start_values, torch.squeeze(self.linear(lstm_out_N), dim=1)], dim=1
            )

        return start_values[:, -num_predictions:]

    def train_forward(
        self, X: torch.Tensor, num_predictions
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # x [N, L]
        mem_predictions, hidden = self.build_memory(X)
        return mem_predictions, self.predict(
            mem_predictions[:, -self.tcn.field_of_view():], hidden, num_predictions
        )

    # x [N, L]
    def forward(self, X: torch.Tensor, num_predictions) -> torch.Tensor:
        predictions, hidden = self.build_memory(X)
        return self.predict(
            predictions[:, -self.tcn.field_of_view():], hidden, num_predictions
        )
