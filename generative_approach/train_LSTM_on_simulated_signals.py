"""
Train the LSTM model on the train dataset split on simulated signals.
"""
from models import CustomLSTM
from mlflow.entities.run_status import RunStatus
import holoviews as hv
from os import makedirs
from pathlib import Path
from sys import maxsize

import mlflow
import torch
from localdataset.dataloader import SharedMemoryLoader
from localdataset.dataset import LocalDataset

from generative_approach.dataset import AllSimulatedMissionsDataSet

mlflow.set_tracking_uri("http://127.0.0.1:8080")


hv.extension("bokeh")


num_memory_build = 750
lstm_hidden_size = 300
num_lstm_layers = 2
num_predictions = 500
bidirectional = True
epochs = 1000
batch_size = 500
learning_rate = 10 ^ -4


if __name__ == "__main__":

    ds = LocalDataset()
    ds.switch_database("simulated_missions")

    missions_data = ds.get_data(
        "None",
        labels=["test_missions"],
        retrievel_class=SharedMemoryLoader,
        lightweight=True,
    ).data

    try:
        # get setup DataLoader
        train_loader = torch.utils.data.DataLoader(
            AllSimulatedMissionsDataSet(
                missions_data, num_memory_build, num_predictions
            ),
            shuffle=True,
            batch_size=batch_size,
            num_workers=8,
            drop_last=True,
        )

        # init Model, optimizer and loss function
        model = CustomLSTM(lstm_hidden_size, num_lstm_layers,
                           bidirectional).to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss(reduction="mean").to("cuda")

        # tmp folder for model states
        tmp_model_dir = Path("/tmp/bckm/models_states")
        makedirs(tmp_model_dir, exist_ok=True)
        experiment = mlflow.set_experiment(
            experiment_name='LSTM-Simulated-Missions-Reduction"MEAN"'
        )

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
                    memory_loss = []
                    hidden = None
                    for i in range(0, num_memory_build - 1):
                        output, hidden = model(
                            torch.unsqueeze(X[:, i], dim=1),
                            (
                                [i.detach() for i in hidden]
                                if hidden is not None
                                else None
                            ),
                        )
                        loss = loss_fn(
                            output, torch.unsqueeze(X[:, i + 1], dim=1))

                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()

                        memory_loss.append(loss.item())

                    # get predictions
                    predictions, hidden = model(
                        torch.unsqueeze(X[:, -1], dim=1),
                        [i.detach() for i in hidden],
                    )
                    for i in range(0, num_predictions - 1):
                        output, hidden = model(
                            torch.unsqueeze(predictions[:, i].detach(), dim=1).to(
                                "cuda"
                            ),
                            hidden,
                        )

                        predictions = torch.concat(
                            [predictions, output], dim=1)

                    loss = loss_fn(predictions, y.to("cuda"))

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # save model states
                    if loss.item() < min_loss:
                        min_loss = loss.item()
                        # save model state dict in mlflow
                        torch.save(
                            model.state_dict(),
                            tmp_model_dir / "model_parameters.pt",
                        )
                        mlflow.log_artifact(
                            str(tmp_model_dir / f"model_parameters.pt"))

                    # log model performance
                    m_b_loss = torch.tensor(memory_loss).mean().sqrt()
                    mlflow.log_metrics(
                        {
                            "training_memory_build_loss": m_b_loss.item(),
                            "training_prediction_loss": loss.sqrt().item(),
                        },
                        step=current_training_batch * batch_size,
                    )

                    if loss.isnan():
                        raise Exception("Prediction loss is 'nan'")
                    elif m_b_loss.isnan():
                        raise Exception("Memory build loss is 'nan'")

    except Exception as e:
        mlflow.set_tag("error", str(e))
        mlflow.end_run(RunStatus.to_string(RunStatus.FAILED))

        print(e)
