"""
Test the trained CNN-Transformer model on the test dataset split.
"""

import json
from argparse import ArgumentParser
from os import makedirs

import mlflow
import numpy as np
import pandas as pd
import torch
from dataset import LocalDataSetWorker
from evaluation_metrics import ClassificationMetrics
from localdataset.dataset import LocalDataset
from model import CNNTransformer
from tqdm import tqdm

mlflow.set_tracking_uri("http://127.0.0.1:8080")

# evaluation metrics

if __name__ == "__main__":
    parser = ArgumentParser(
        prog="Script to measure the performance of the CNN-Transformer model")

    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to run model on. Default options: cuda, cpu",
    )

    parser.add_argument(
        "-p",
        "--preprocessing",
        nargs="*",
        default=["raw"],
        type=str,
        help="If preprocessing is not defined, script uses default 'base' preprocessing.",
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

    if args.mlflow_tracking_uri is not None:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

    # get mlflow run
    run = mlflow.get_run(args.run_id)
    run_params = run.data.params

    # build model according to run and use runs parameters
    model = CNNTransformer(
        int(run_params["d_model"]),
        int(run_params["stride"]),
        int(run_params["kernel_size"]),
        int(run_params["nhead"]),
        int(run_params["num_layers"]),
        float(run_params["width_multiplier"]),
        float(run_params["dropout"]),
        json.loads(run_params["FFN_dim_hidden_layers"])[:-1],
    ).to(device)
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

    for p in args.preprocessing:
        # get setup DataLoader
        data = ds.get_data(
            preprocessing=p,
            lightweight=True,
            labels=["test_mission", "test_missions",
                    "real_mission", "real_missions"],
        )

        loader = data.test.get_by_index
        map_labels = {  # necessary because some missions where not labeled correctly
            "test_mission": 0,
            "test_missions": 0,
            "real_mission": 1,
            "real_missions": 1,
        }

        test_loader = torch.utils.data.DataLoader(
            LocalDataSetWorker(loader, data.test.indices, map_labels, True),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=8,
            drop_last=True,
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

        print(pd.DataFrame.from_records([performance_metric]))

        df.to_pickle(f"{args.output}/classification_results.pkl")
