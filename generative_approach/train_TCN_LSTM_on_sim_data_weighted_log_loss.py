"""
Train TCN-LSTM Model on all simulated missions using the weighted log loss
"""

from os import makedirs
from pathlib import Path
from sys import maxsize

import mlflow
import torch
from localdataset.dataloader import SharedMemoryLoader
from localdataset.dataset import LocalDataset
from models import TCN_LSTM, CustomLSTM
from tqdm import tqdm

from generative_approach.loss_functions import WeightedMSELoss

mlflow.set_tracking_uri("http://127.0.0.1:8080")


rate = 300
num_memory_build = 5 * rate
num_predictions = 5 * rate

bidirectional = True

lstm_hidden_size = 64
num_lstm_layers = 2
conv_channels = 64
kernel_size = 3

epochs = 5
batch_size = 400

learning_rate = 0.00001

loss_scale = 1000000

s_len = 20000


if __name__ == "__main__":

    ds = LocalDataset()
    ds.switch_database("simulated_missions")

    missions_data = ds.get_data(
        "None",
        labels=["test_mission", "test_missions"],
        retrievel_class=SharedMemoryLoader,
        lightweight=True,
    ).data

    # get train data and setup DataLoader
    train_loader = torch.utils.data.DataLoader(
        CustomLSTM(missions_data, num_memory_build, num_predictions),
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
    )

    # tmp folder for model states
    tmp_model_dir = Path("/tmp/bckm/models_states")
    makedirs(tmp_model_dir, exist_ok=True)

    device = torch.device("cuda")

    # init Model, optimizer and loss function
    model = TCN_LSTM(
        conv_channels, kernel_size, lstm_hidden_size, num_lstm_layers, bidirectional
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_weight_mse = WeightedMSELoss(10, num_predictions, loss_scale).to(
        device
    )  # training loss function

    loss_mse = torch.nn.MSELoss().to(device)  # for performance reports

    experiment = mlflow.set_experiment(
        experiment_name="LSTM-end-to-end-training-test-Real-Data"
    )

    min_prediction_loss = maxsize
    with mlflow.start_run(
        run_name=f"TCN_LSTM_{loss_scale}-{num_memory_build}-{num_predictions}-{lstm_hidden_size}-{num_lstm_layers}{bidirectional}-retry"
    ):
        # log params
        mlflow.log_params(
            {
                "num_memory_build": num_memory_build,
                "num_predictions": num_predictions,
                "lstm_hidden_size": lstm_hidden_size,
                "num_lstm_layers": num_lstm_layers,
                "epochs": epochs,
                "batch_size": batch_size,
                "bidirectional": bidirectional,
                "learning_rate": learning_rate,
                "conv_channels": conv_channels,
                "kernel_size": kernel_size,
                "loss_scale": loss_scale,
            }
        )

        current_training_batch = 0

        fov = model.tcn.field_of_view()

        for e in tqdm(range(epochs)):
            for X, y in train_loader:  # tain model
                X = X.to(device).float()
                y = y.to(device).float()

                # build memory
                memory_losses = []
                hidden = None
                for i in range(X.shape[1] - fov):
                    c = model.tcn(torch.unsqueeze(X[:, i: i + fov], dim=1))
                    out, hidden = model.lstm(
                        torch.unsqueeze(c[:, :, 0], dim=1),
                        [i.detach()
                         for i in hidden] if hidden is not None else None,
                    )
                    prediction = torch.squeeze(
                        model.linear(torch.squeeze(out, dim=1)), dim=1
                    )

                    target = X[:, i + fov]
                    loss = loss_mse(target, prediction)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    memory_losses.append(loss.item())

                # predict

                predictions = model.predict(
                    X[:, -fov:], [i.detach() for i in hidden], num_predictions
                )

                prediction_loss = loss_weight_mse(predictions, y)

                losses = {
                    f"loss-{i}": loss_mse(
                        predictions[:, : (i + 1) * 20], y[:, : (i + 1) * 20]
                    )
                    for i in range(int(num_predictions / 20))
                }

                # save model states
                if prediction_loss.item() < min_prediction_loss:
                    min_prediction_loss = prediction_loss.item()
                    # save model state dict in mlflow
                    torch.save(
                        model.state_dict(), tmp_model_dir / "model_parameters.pt"
                    )
                    mlflow.log_artifact(
                        str(tmp_model_dir / f"model_parameters.pt"))

                # log model performance
                losses.update(
                    {
                        "training_memory_build_loss": torch.tensor(memory_losses)
                        .mean()
                        .item(),
                        "training_prediction_loss": prediction_loss.item(),
                    }
                )
                mlflow.log_metrics(losses, step=current_training_batch)
                current_training_batch += 1
