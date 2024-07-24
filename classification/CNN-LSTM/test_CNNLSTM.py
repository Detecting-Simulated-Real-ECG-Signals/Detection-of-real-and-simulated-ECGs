"""
Test the trained CNN-LSTM model on the test dataset split.
"""

from tqdm import tqdm
from sklearn.metrics import auc, roc_curve
from model import CNNLSTM
from localdataset.dataset import LocalDataset
from evaluation_metrics import ClassificationMetrics
from dataset import LocalDataSetWorker
import torch
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
from os import makedirs
from argparse import ArgumentParser

mlflow.set_tracking_uri("http://127.0.0.1:8080")


if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Script to measure the performance of the CNN-LSTM model")

    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to run the model on. Default options: cuda, cpu"
    )
    parser.add_argument(
        "-p",
        "--preprocessing",
        nargs="*",
        default=["raw"],
        type=str,
        help="If preprocessing is not defined, script uses default 'base' preprocessing."
    )
    parser.add_argument(
        "-d",
        "--database",
        default=None,
        type=str,
        help="The used database must contain the preprocessing selected with '--preprocessing'. This argument requires a path to the preprocessing files.",
    )
    parser.add_argument("-r", "--run-id", required=True,
                        type=str, help="MLflow run id")
    parser.add_argument(
        "-o", "--output", default=None, type=str, help="default: 'output/{run-id}'"
    )
    parser.add_argument("--num-workers", default=8, type=int,
                        help="Number of workers used to preprocess data.")

    parser.add_argument("-b", "--batch-size", default=1,
                        type=int, help="Batch size")

    parser.add_argument("--mlflow-tracking-uri",
                        default=None, help="MLflow tracking uri")
    parser.add_argument("--mlflow-experiment",
                        default="Classification_CNN-Transformer", help="MLflow experiment name")

    args = parser.parse_args()

    device = torch.device(args.device)

    print(device)

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # get mlflow run
    run = mlflow.get_run(args.run_id)
    run_prarms = run.data.params

    # build model according to run and use runs parameters
    model = CNNLSTM(bool(run_prarms["bidirektional"])).to(device)
    model.load_state_dict(
        torch.load(
            mlflow.artifacts.download_artifacts(
                artifact_uri=str(run.info.artifact_uri) +
                "/model_parameters.pt"
            ),
            map_location=device,
        )
    )
    model.eval()

    softmax = torch.nn.Softmax(1).to(device)

    # setup output path
    if args.output is None:
        args.output = f"output/{args.run_id}"

    makedirs(args.output)

    # load dataset
    ds = LocalDataset("/media/HDD01/mbck/dataSet/")

    if args.database is not None:
        ds.switch_database(path=args.database)

    # classification metrics
    classification_metrics = ClassificationMetrics().to(device)

    performance_metrics = []

    # create figure
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)

    # add line
    ax.plot([0, 1], [0, 1], color="#CCCCCC", linestyle="dashed")

    # get setup DataLoader
    for p in args.preprocessing:
        data = ds.get_data(
            preprocessing=p,
            lightweight=True,
            labels=["test_mission", "test_missions",
                    "real_mission", "real_missions"],
        )

        loader = data.test.get_by_index
        map_labels = {
            "test_mission": 0,
            "test_missions": 0,
            "real_mission": 1,
            "real_missions": 1,
        }

        test_loader = torch.utils.data.DataLoader(
            LocalDataSetWorker(loader, data.test.indices, map_labels),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=8,
            drop_last=False,
        )

        results = []

        with torch.no_grad():
            for X, y in tqdm(test_loader):
                X = torch.unsqueeze(X, dim=1).to(device)

                # predict
                prediction = model(X)
                s_predictions = softmax(prediction)

                # evaluate
                metrics = classification_metrics(s_predictions, y.to(device))
                metrics["prediction"] = s_predictions.cpu()[0].tolist()
                metrics["y"] = y.cpu()[0].tolist()

                results.append(metrics)

        df = pd.DataFrame.from_records(results)

        y = np.array(df["y"].values.tolist()).argmax(axis=1)
        prediction = np.array(df["prediction"].values.tolist())

        fpr, tpr, thresholds = roc_curve(y, np.abs(prediction[:, 0] - 1))
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"AUC = {roc_auc*100:2.2f}%")

        data = np.array(
            [
                (p[0], p[1], y[0], y[1])
                for p, y in zip(df["prediction"].values, df["y"].values)
            ]
        )
        prediction, target = torch.from_numpy(data[:, :2]), torch.from_numpy(
            data[:, 2:]
        )

        performance_metric = ClassificationMetrics()(prediction, target)
        performance_metric["snippet-size"] = p
        performance_metrics.append(performance_metric)

    print(pd.DataFrame.from_records(performance_metrics))

    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_ylabel("True Postive Rate")
    ax.set_xlabel("False Postive Rate")
    ax.legend(loc="lower right")

    fig.savefig(f"{args.output}/roc-curve.png", dpi=1000)
    print(f"\nOutput: {args.output}/roc-curve.png")
