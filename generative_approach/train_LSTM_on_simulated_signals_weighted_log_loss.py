"""
Train the LSTM model on the train dataset split using the custom loss with the weighted log loss.
"""
from os import makedirs
from pathlib import Path
from sys import maxsize

import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from localdataset.dataloader import SharedMemoryLoader
from localdataset.dataset import LocalDataset
from loss_functions import WeightedMSELoss
from models import CustomLSTM
from tqdm import tqdm

from generative_approach.dataset import AllSimulatedMissionsDataSet

mlflow.set_tracking_uri("http://127.0.0.1:8080")


num_memory_build = 1000
num_predictions = 400

bidirectional = True

lstm_hidden_size = 300
num_lstm_layers = 5

epochs = 100
batch_size = 450

learning_rate = 0.00001

loss_scale = 1000000

s_len = 20000


if __name__ == "__main__":

    ds = LocalDataset()
    ds.switch_database("simulated")

    missions_data = ds.get_data(
        "None",
        labels=["test_mission", "test_missions"],
        retrievel_class=SharedMemoryLoader,
        lightweight=True,
    ).data

    # get train data and setup DataLoader
    train_loader = torch.utils.data.DataLoader(
        AllSimulatedMissionsDataSet(
            missions_data, num_memory_build, num_predictions),
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
    )

    device = torch.device("cuda")

    # init Model, optimizer and loss function
    model = CustomLSTM(lstm_hidden_size, num_lstm_layers,
                       bidirectional).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss().to(device)
    loss_weight_mse = WeightedMSELoss(
        10, num_predictions, loss_scale).to(device)
    loss_mse = torch.nn.MSELoss().to(device)

    # tmp folder for model states

    tmp_model_dir = Path("/tmp/bckm/models_states")
    makedirs(tmp_model_dir, exist_ok=True)

    # tmp folder for model states
    tmp_model_dir = Path("/tmp/bckm/models_states")
    makedirs(tmp_model_dir, exist_ok=True)
    experiment = mlflow.set_experiment(
        experiment_name="LSTM-end-to-end-training-test-Real-Data"
    )

    min_loss = maxsize
    with mlflow.start_run(
        run_name=f"weighted_log_loss_{loss_scale}-{num_memory_build}-{num_predictions}-{lstm_hidden_size}-{num_lstm_layers}{bidirectional}-retry"
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
            }
        )

        current_training_batch = 0

        for e in tqdm(range(epochs)):
            for X, y in train_loader:  # tain model
                X = X.to(device).float()
                y = y.float()

                # build memory
                memory_loss = []
                hidden = None
                for i in range(0, num_memory_build - 1):
                    output, hidden = model(
                        torch.unsqueeze(X[:, i], dim=1),
                        [i.detach()
                         for i in hidden] if hidden is not None else None,
                    )
                    loss = loss_mse(
                        output, torch.unsqueeze(X[:, i + 1], dim=1))
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()

                    memory_loss.append(loss.item())

                # get predictions
                predictions, hidden = model(
                    torch.unsqueeze(X[:, -1], dim=1), [i.detach()
                                                       for i in hidden]
                )
                for i in range(0, num_predictions - 1):
                    output, hidden = model(
                        torch.unsqueeze(
                            predictions[:, i].detach(), dim=1).to(device),
                        hidden,
                    )

                    predictions = torch.concat([predictions, output], dim=1)

                y = y.to(device)
                loss = loss_weight_mse(predictions, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses = {
                    f"loss-{i}": loss_mse(predictions[:, : i * 20], y[:, : i * 20])
                    for i in range(int(num_predictions / 20))
                }

                # save model states
                if loss.item() < min_loss:
                    min_loss = loss.item()
                    # save model state dict in mlflow
                    torch.save(
                        model.state_dict(), tmp_model_dir / "model_parameters.pt"
                    )
                    mlflow.log_artifact(
                        str(tmp_model_dir / f"model_parameters.pt"))

                # log model performance
                losses.update(
                    {
                        "training_memory_build_loss": torch.tensor(memory_loss)
                        .mean()
                        .item(),
                        "training_prediction_loss": loss.item(),
                    }
                )
                mlflow.log_metrics(losses, step=current_training_batch)
                current_training_batch += 1
