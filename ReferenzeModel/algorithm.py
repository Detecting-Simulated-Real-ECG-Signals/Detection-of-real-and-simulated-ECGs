"""
Apply the similarity based Algorithm to classify missions using the threshold classification approach.
"""

from itertools import count
from os import makedirs
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from localdataset.dataset import LocalDataset
from tqdm import tqdm

from ReferenzeModel.tools import (MissionCorrelationV2, MissionWindowing,
                                  correlation)


def calc_mission_correlation(
    mission: np.ndarray,
    left_window_size: int,
    max_steps: int,
    stride: int,
    threshold: float,
) -> Dict[Tuple[int, int], int]:
    mw = MissionWindowing(mission, left_window_size, max_steps, stride)

    correlation_streak = {}
    for j, l_w in enumerate(mw):
        try:
            for i in count(0, 1):
                correlation_streak[(j, i)] = []
                r_w = mw.next_right_window()
                c = correlation(l_w, r_w)
                # print(f" {c} {l_w} {r_w}")

                try:
                    while c > threshold:
                        correlation_streak[(j, i)].append(c)

                        l_w = mw.next_left_window()
                        r_w = mw.next_right_window()
                        c = correlation(l_w, r_w)
                except StopIteration as e:
                    pass
        except StopIteration as e:
            pass
    return correlation_streak


def mission_normalized(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError("only 1D array supported")
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values  # , run_starts, run_lengths


def calc_mission_correlation_coefficients(
    mission: np.ndarray,
    mission_correlations: Dict[str, List[float]],
    classify_threshold: float,
    reduction=sum,
    normalization=True,
):

    if normalization:
        mission_repeation = (
            reduction(
                [len(v) if v is not None else 0 for v in mission_correlations.values()]
            )
            / mission_normalized(mission).shape[0]
            if len(mission_correlations.values()) > 0
            else 0
        )
    else:
        mission_repeation = (
            reduction(
                [len(v) if v is not None else 0 for v in mission_correlations.values()]
            )
            if len(mission_correlations.shape[0]) > 0
            else 0
        )

    min_prediction_value = 0
    max_prediction_value = 0.038569206842923795

    prediction = np.clip(
        np.array(mission_repeation), min_prediction_value, max_prediction_value
    )

    if prediction >= classify_threshold:
        percentage = (
            (prediction - classify_threshold)
            / (max_prediction_value - classify_threshold)
        ) * 0.5 + 0.5
    else:
        percentage = (
            (prediction) / (min_prediction_value + classify_threshold)) * 0.5

    return percentage


if __name__ == "__main__":

    # Parameters
    left_window_size = 300
    right_window_size = 7000
    max_steps = 9300
    stride = 300
    parallel_correlations = 100
    window_stride = left_window_size

    def reduction_method(x): return sum(x) / len(x)

    mission_len_normalization = True

    similarity_threshold = 0.905
    classification_threshold = 0.0009874704422230714

    database = "database-path"
    output_folder = "output-folder"

    correlation_class = MissionCorrelationV2(
        left_window_size, max_steps, stride, similarity_threshold
    )

    # load data
    ds = LocalDataset()

    # ds.switch_database(path=database)

    data = ds.get_data(
        labels=["test_mission", "real_mission"],
        lightweight=True,
    )

    test_indices = data.test.indices
    loader = data.test.get_by_index

    map_labels = {"test_mission": 1, "real_mission": 0}

    # calculate
    predictions = []

    for index in tqdm(test_indices):
        X, label = loader(index)

        mission_correlations = {
            k: v for k, v in correlation_class.calc(X).items() if len(v) > 0
        }
        prediction = float(
            calc_mission_correlation_coefficients(
                X,
                mission_correlations,
                classification_threshold,
                reduction_method,
                mission_len_normalization,
            )
        )

        y = [0, 0]
        y[map_labels[label]] = 1
        p = [1 - prediction, prediction]

        predictions.append({"prediction": p, "y": y})

    makedirs(output_folder, exist_ok=True)
    pd.DataFrame(predictions).to_pickle(f"{output_folder}/results.pkl")
