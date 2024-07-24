"""
Train the CNN-Transformer model on the train dataset split and validate it on the validation split.
"""

from argparse import ArgumentParser
from os import makedirs
from pathlib import Path

import mlflow
import torch
from dataset import LocalDataSetWorker
from evaluation_metrics import ClassificationMetrics
from localdataset.dataset import LocalDataset
from model import CNNTransformer

mlflow.set_tracking_uri("http://127.0.0.1:8080")

# evaluation metrics

if __name__ == "__main__":
    parser = ArgumentParser(prog="Train the CNN-Transformer model")

    parser.add_argument("-p", "--preprocessing", required=True, type=str,
                        help="If preprocessing is not defined, script uses default 'base' preprocessing.",)

    parser.add_argument("--device", default="cuda", type=str,
                        help="Device to run the model on. Default options: cuda, cpu")
    parser.add_argument("-lr", "--learning-rate", default=0.0001,
                        type=float, help="Learning rate used to optimize the model.")

    parser.add_argument("--d_model", default=64, type=int,
                        help="d_model size of the transformer model.")
    parser.add_argument("--stride", default=2, type=int,
                        help="Stride used by the CNN backbone.")
    parser.add_argument("--kernel_size", default=3, type=int,
                        help="Kernal size of the CNN backbone.")
    parser.add_argument("--nhead", default=8, type=int,
                        help="Number of multi headed attention used by the transformer.")
    parser.add_argument("--num_layers", default=6, type=int,
                        help="Number of Transformer layers.")
    parser.add_argument("--width_multiplier", default=0.25, type=float,
                        help="Channel multiplier used by the CNN backbone.")
    parser.add_argument("--dropout", default=0.1,
                        type=float, help="Dropout percentage.")

    parser.add_argument(
        "--FFN_dim_hidden_layers", default=[], nargs="*", action="append", type=int,
        help="Number of neurons used by the classification component used as hidden layer"
    )

    parser.add_argument("--validation-batches", default=30, type=int,
                        help="Amount of validation batches used to validate the models performance.")

    parser.add_argument("--epochs", default=1, type=int,
                        help="Number of training epochs.")
    parser.add_argument("--num-workers", default=8, type=int,
                        help="Number of workers used to preprocess data.")

    parser.add_argument("-b", "--batch-size", default=1,
                        type=int, help="Batch size.")

    parser.add_argument("--mlflow-tracking-uri",
                        default=None, help="MLflow tracking uri.")
    parser.add_argument("--mlflow-experiment",
                        default="Classification_CNN-Transformer", help="MLflow experiment name.")

    args = parser.parse_args()

    device = torch.device(args.device)

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    ffn_hidden_layers = []
    for i in args.FFN_dim_hidden_layers:
        ffn_hidden_layers.extend(i)

    model = CNNTransformer(
        args.d_model,
        args.stride,
        args.kernel_size,
        args.nhead,
        args.num_layers,
        args.width_multiplier,
        args.dropout,
        ffn_hidden_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = torch.nn.BCELoss().to(device)

    softmax = torch.nn.Softmax(1).to(device)

    # load dataset
    ds = LocalDataset("/media/HDD01/mbck/dataSet/")

    # get setup DataLoader
    data = ds.get_data(
        lightweight=True,
        preprocessing=args.preprocessing,
        labels=["test_mission", "test_missions",
                "real_mission", "real_missions"],
    )

    loader = data.train.get_by_index

    map_labels = {
        "test_mission": 0,
        "test_missions": 0,
        "real_mission": 1,
        "real_missions": 1,
    }

    train_loader = torch.utils.data.DataLoader(
        LocalDataSetWorker(loader, data.train.indices, map_labels),
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
    )
    validation_loader = torch.utils.data.DataLoader(
        LocalDataSetWorker(loader, data.validation.indices, map_labels),
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=8,
        drop_last=True,
    )

    classification_metrics = ClassificationMetrics().to(device)

    # create tmp folder for model parameters
    tmp_model_dir = Path("/tmp/bckm/models_states")
    makedirs(tmp_model_dir, exist_ok=True)

    experiment = mlflow.set_experiment(
        experiment_name="Classification_CNN-Transformer")

    with mlflow.start_run(
        run_name=f"{args.preprocessing}-FFN{'-'.join([str(i) for i in ffn_hidden_layers])}"
    ):
        mlflow.log_params(
            {
                "preprocessing": args.preprocessing,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "d_model": args.d_model,
                "stride": args.stride,
                "kernel_size": args.kernel_size,
                "nhead": args.nhead,
                "num_layers": args.num_layers,
                "width_multiplier": args.width_multiplier,
                "dropout": args.dropout,
                "FFN_dim_hidden_layers": ffn_hidden_layers,
                "learning_rate": args.learning_rate,
                "validation-batches": args.validation_batches,
            }
        )

        batch = 0
        min_val_loss = 100

        for e in range(args.epochs):
            for X, y in train_loader:
                model = model.train()
                batch += 1
                X = torch.unsqueeze(X, dim=1).to("cuda")
                y = y.to("cuda")

                # build memory
                prediction = model(X)
                s_predictions = softmax(prediction)

                loss = loss_fn(s_predictions, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                mlflow.log_metric("loss", loss.item(), step=batch)
                mlflow.log_metrics(classification_metrics(
                    s_predictions, y), step=batch)

                # evaluate models performance
                if batch % 10 == 0:

                    val_loader = iter(validation_loader)

                    val_results = {
                        "loss": [],
                        "Accuracy": [],
                        "Specificity": [],
                        "Precision": [],
                        "F1": [],
                        "Recall": [],
                    }

                    model = model.eval()
                    with torch.no_grad():
                        for i in range(args.validation_batches):
                            X_val, y_val = next(val_loader)
                            X_val = torch.unsqueeze(X_val, dim=1).to("cuda")
                            y_val = y_val.to("cuda")

                            prediction = model(X_val)
                            s_predictions = softmax(prediction)

                            val_results["loss"].append(
                                loss_fn(s_predictions, y_val).item()
                            )

                            val_metrics = classification_metrics(
                                s_predictions, y_val)

                            val_results["Accuracy"].append(
                                val_metrics["Accuracy"])
                            val_results["Specificity"].append(
                                val_metrics["Specificity"]
                            )
                            val_results["Precision"].append(
                                val_metrics["Precision"])
                            val_results["F1"].append(val_metrics["F1"])
                            val_results["Recall"].append(val_metrics["Recall"])

                    v_loss = sum(val_results["loss"]) / args.validation_batches
                    if v_loss < min_val_loss:
                        min_loss = v_loss
                        # save model state dict in mlflow
                        torch.save(
                            model.state_dict(), tmp_model_dir / "model_parameters.pt"
                        )
                        mlflow.log_artifact(
                            str(tmp_model_dir / f"model_parameters.pt"))

                    total_val_metrics = {
                        f"validation_{k}": sum(v) / args.validation_batches
                        for k, v in val_results.items()
                    }

                    val_losses = torch.tensor(val_results["loss"])
                    total_val_metrics["validation_min_loss"] = val_losses.min(
                    ).item()
                    total_val_metrics["validation_max_loss"] = val_losses.max(
                    ).item()

                    mlflow.log_metrics(total_val_metrics, step=batch)

                    if batch > 10000:
                        exit()
