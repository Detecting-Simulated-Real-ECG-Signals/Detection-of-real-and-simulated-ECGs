"""
Train the Encoder-Decoder LSTM model on the train dataset split using the custom loss with dynamic prediction size.
"""
from os import makedirs
from pathlib import Path
from sys import maxsize

import mlflow
import torch
from localdataset.dataloader import SharedMemoryLoader
from localdataset.dataset import LocalDataset
from models import EncoderDecoderLSTM

from generative_approach.dataset import AllSimulatedMissionsDataSet

mlflow.set_tracking_uri("http://127.0.0.1:8080")

# hyperparameters
num_memory_build = 500
num_predictions = 1000

bidirectional = True

lstm_hidden_size = 300
num_lstm_layers = 5

epochs = 100
batch_size = 150

learning_rate = 0.001


if __name__ == "__main__":
    ds = LocalDataset()
    ds.switch_database("simulated_missions")

    missions_data = ds.get_data(
        "None",
        labels=["test_mission", "test_missions"],
        retrievel_class=SharedMemoryLoader,
        lightweight=True,
    ).data

    train_loader = torch.utils.data.DataLoader(
        AllSimulatedMissionsDataSet(
            missions_data, num_memory_build, num_predictions),
        shuffle=True,
        batch_size=batch_size,
        num_workers=8,
        drop_last=True,
    )

    # init Model, optimizer and loss function
    model = EncoderDecoderLSTM(lstm_hidden_size, num_lstm_layers, bidirectional).to(
        "cuda"
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss(reduction="sum").to("cuda")

    # get setup DataLoader

    # tmp folder for model states
    tmp_model_dir = Path("/tmp/bckm/models_states")
    makedirs(tmp_model_dir, exist_ok=True)
    experiment = mlflow.set_experiment(
        experiment_name="EncoderDecoderLSTM-Training")

    min_loss = maxsize
    with mlflow.start_run(
        run_name=f"{num_memory_build}-{num_predictions}-{lstm_hidden_size}-{num_lstm_layers}{bidirectional}"
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

        for e in range(epochs):
            for X, y in train_loader:
                current_training_batch += 1
                X = X.to("cuda").float()
                y = y.to("cuda").float()

                # build memory
                prediction = model(X, num_predictions)

                loss = loss_fn(prediction, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

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
                mlflow.log_metrics(
                    {"loss": loss.sqrt().item()}, step=current_training_batch
                )
