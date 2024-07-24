"""
Base module containing both Algorithm V1 and V2 and additionally a parallel execution method.
"""

import json
from collections import defaultdict
from itertools import count
from math import ceil
from multiprocessing import Pool, Queue
from time import sleep
from typing import Dict, Generator, List, Tuple

import numpy as np
from localdataset.dataset import relevant_mission
from tqdm import tqdm


def padding(X: np.ndarray, right_padding) -> np.ndarray:
    return np.pad(X, pad_width=(0, right_padding))


def windowing(
    mission: np.ndarray, left_window_size, right_window_size, stride
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """create sliding window"""

    # pad mission
    w_size = left_window_size + right_window_size

    mission = np.pad(mission, pad_width=(0, left_window_size -
                                         ((mission.shape[0] - right_window_size) % left_window_size)))

    for i in range(0, int(mission.shape[0] - left_window_size * 2), stride):
        yield (i, mission[i: i + w_size])


def correlation(x: np.ndarray, y: np.ndarray) -> float:
    return np.corrcoef(x, y)[0, 1]


def autocorrelation(X: np.ndarray, parallel_correlations=1) -> np.ndarray:
    """Calc autocorrelation on X."""
    X = X + np.finfo(np.float32).eps

    correlations = np.zeros(X.shape)

    for j in range(ceil((X.shape[0] - 1) / parallel_correlations)):
        lower_bound = j * parallel_correlations + 1
        upper_bound = min([(j + 1) * parallel_correlations + 1, X.shape[0]])

        correlations[lower_bound:upper_bound] = np.corrcoef(
            np.stack(
                [X, *[np.roll(X, shift=i)
                      for i in range(lower_bound, upper_bound)]]
            )
        )[0, 1:]

    return correlations


class Correlation:
    """
    Correlation Class.
    """

    def calc(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:  # , parallel_correlations=1) -> np.ndarray:
        """Calc correlation between X and y."""
        assert X.shape[0] <= y.shape[0], "X must have a smaller shape then y!"

        machine_epsilon = np.finfo(np.float32).eps
        X = X + machine_epsilon
        y = y + machine_epsilon

        correlations = np.zeros((y.shape[0] - X.shape[0]) + 1)

        for i in range((y.shape[0] - X.shape[0]) + 1):
            correlations[i] = np.corrcoef(X, y[i: i + X.shape[0]])[0, 1]

        return correlations


class CorrelationThreshold(Correlation):
    """
    Correlation Class.
    """

    def __init__(self, threshold: float = 0.92) -> None:
        super().__init__()
        self.threshold = threshold

    def calc(
        self, X: np.ndarray, y: np.ndarray
    ) -> np.ndarray:  # , parallel_correlations=1) -> np.ndarray:
        """Calc correlation between X and y."""
        assert X.shape[0] <= y.shape[0], "X must have a smaller shape then y!"

        machine_epsilon = np.finfo(np.float32).eps
        X = X + machine_epsilon
        y = y + machine_epsilon

        correlations = np.zeros((y.shape[0] - X.shape[0]) + 1)

        for i in range((y.shape[0] - X.shape[0]) + 1):
            c = np.corrcoef(X, y[i: i + X.shape[0]])[0, 1]
            if c > self.threshold:
                correlations[i] = c

        return correlations


def check_window_relevance(window, left_window_size, threshold) -> bool:
    # right_window_size = window.shape[0] - left_window_size
    left_window_relevant = relevant_mission(
        window[:left_window_size], threshold=threshold
    )
    return left_window_relevant  # and right_window_relevant


class MissionCorrelation:
    '''

    '''

    def __init__(
        self,
        left_window_size: int,
        right_window_size: int,
        window_stride: float,
        correlation_class: Correlation = Correlation(),
    ):
        self.left_window_size = left_window_size
        self.right_window_size = right_window_size
        self.window_stride = window_stride
        self.correlation_class = correlation_class

    def calc(self, mission_index, mission):
        c_results: List[Tuple[np.int_, np.float_]] = []
        for window_start_index, window in windowing(
            mission, self.left_window_size, self.right_window_size, self.window_stride
        ):
            if check_window_relevance(window, self.left_window_size, 0.1):
                X = window[: self.left_window_size]
                y = window[self.left_window_size:]
                c_res = self.correlation_class.calc(X, y)
                c_results.extend(
                    [(window_start_index + i, c) for i, c in enumerate(c_res)]
                )

        return c_results


class MissionCorrelationDensity(MissionCorrelation):
    def calc(self, mission_index, mission):
        c_results: Dict[int, List[np.float_]] = defaultdict(list)
        for window_start_index, window in windowing(
            mission, self.left_window_size, self.right_window_size, self.window_stride
        ):
            if check_window_relevance(window, self.left_window_size, 0.1):
                X = window[: self.left_window_size]
                y = window[self.left_window_size:]
                c_res = self.correlation_class.calc(X, y)
                for i, c in enumerate(filter(lambda x: x > 0, c_res)):
                    c_results[window_start_index + i].append(c)

        siginficant_correlations = np.zeros(mission.shape)
        for k, v in c_results.items():
            siginficant_correlations[k] = max(v)

        return siginficant_correlations


class MissionWindowing:
    def __init__(
        self, mission: np.ndarray, left_window_size: int, max_steps: int, stride: int
    ) -> None:
        self.mission: np.ndarray = mission
        self.left_window_size: int = left_window_size
        self.max_steps: int = max_steps
        self.stride: int = stride

        self.window_index: int = -1
        self.x_offset: int = 0
        self.y_offset: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        if (
            int(self.window_index * self.stride) + 2 * self.left_window_size
            < self.mission.shape[0]
        ):
            return self.next_window()
        else:
            raise StopIteration

    def __len__(self) -> int:
        return int(self.mission.shape[0] / self.stride)

    def next_window(self) -> np.ndarray:
        self.x_offset = 0
        self.y_offset = 0
        self.window_index += 1

        start_index = int(self.window_index * self.stride)

        return self.mission[start_index: start_index + self.left_window_size]

    def next_left_window(self) -> np.ndarray:
        start_index = int(self.window_index * self.stride) + \
            (self.y_offset + 1)
        area = self.mission[start_index: start_index + self.left_window_size]
        self.y_offset += 1
        if area.shape[0] >= self.left_window_size:
            return area
        else:
            raise StopIteration("mission out of segments")

    def next_right_window(self) -> np.ndarray:
        if self.x_offset >= self.max_steps:
            raise StopIteration("reached max steps.")

        start_index = (
            int(self.window_index * self.stride) +
            self.left_window_size + self.x_offset
        )

        if start_index + self.left_window_size > self.mission.shape[0]:
            raise StopIteration("mission out of segments.")

        area = self.mission[start_index: start_index + self.left_window_size]
        self.x_offset += 1
        return area


class MissionCorrelationV2(MissionCorrelation):
    def calc(self, mission):
        mw = MissionWindowing(
            mission, self.left_window_size, self.max_steps, self.stride
        )

        correlation_streak = {}
        for j, l_w in enumerate(mw):
            try:
                for i in count(0, 1):
                    correlation_streak[f"{j}_{i}"] = []
                    r_w = mw.next_right_window()
                    c = correlation(l_w, r_w)

                    try:
                        while c > self.threshold:
                            correlation_streak[f"{j}_{i}"].append(float(c))

                            l_w = mw.next_left_window()
                            r_w = mw.next_right_window()
                            c = correlation(l_w, r_w)
                    except StopIteration as e:
                        pass
            except StopIteration as e:
                pass

        return correlation_streak


def get_correlation(
    mission: np.ndarray,
    left_window_size: int,
    right_window_size: int,
    window_stride: float,
    correlation_class: Correlation = Correlation(),
) -> List[Tuple[np.int_, np.float_]]:
    c_results: List[Tuple[np.int_, np.float_]] = []
    for window_start_index, window in windowing(
        mission, left_window_size, right_window_size, window_stride
    ):
        if check_window_relevance(window, left_window_size, 0.1):
            X = window[:left_window_size]
            y = window[left_window_size:]
            c_res = correlation_class.calc(X, y)
            c_results.extend([(window_start_index + i, c)
                             for i, c in enumerate(c_res)])

    return c_results


def get_correlation_density(
    mission: np.ndarray,
    left_window_size: int,
    right_window_size: int,
    window_stride: float,
    correlation_class: Correlation = Correlation(),
    density_window_size: int = 500,
) -> float:
    c_results: List[Tuple[np.int_, np.float_]] = get_correlation(
        mission, left_window_size, right_window_size, window_stride, correlation_class)

    siginficant_correlations = np.zeros(mission.shape)
    for k, v in c_results.items():
        siginficant_correlations[k] = max(v)

    return np.mean(
        np.lib.stride_tricks.sliding_window_view(
            siginficant_correlations, density_window_size
        ),
        axis=1,
    ).max()


def multiprocessing_worker(
    input_queue: Queue,
    result_queue: Queue,
    dataloader,
    correlation_class: MissionCorrelation,
    save_in_file: bool = False,
):
    try:
        while True:
            mission_index = input_queue.get()
            mission_object = dataloader(mission_index)
            data = mission_object.data
            label = mission_object.label
            gid = mission_object.gid

            correlations = correlation_class.calc(mission_index, data)

            if save_in_file:
                with open(
                    f"alg_correlations/correlations/correlation_{mission_index}.json",
                    "w",
                ) as writer:
                    writer.write(json.dumps(correlations))
                result_queue.put({})
            else:
                result_queue.put({mission_index: (gid, label, correlations)})
    except Exception as e:
        pass


def parallel_correlation(
    workers: int,
    loader,
    train_indexes,
    correlation_class: MissionCorrelation,
    save_in_file: bool = False,
):
    result_queue = Queue()
    input_queue = Queue()

    p = Pool(
        workers,
        multiprocessing_worker,
        (input_queue, result_queue, loader, correlation_class, save_in_file),
    )

    for i in train_indexes:
        input_queue.put(i)

    correlations = {}
    for _ in tqdm(range(len(train_indexes))):
        while result_queue.empty():
            sleep(1)
        correlations.update(result_queue.get())

    p.close()

    if not save_in_file:
        return correlations
