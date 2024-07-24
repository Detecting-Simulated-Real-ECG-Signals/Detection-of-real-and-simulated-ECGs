import unittest
from os import mkdir, rmdir
from random import randint

import numpy as np

from ReferenzeModel.tools import MissionCorrelationV2, MissionWindowing


class TestMissionWindowing(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def test_fist_right_window(self):
        l = self.mw.next_window()
        r = self.mw.next_right_window()
        self.assertTrue(
            l.tolist() +
            r.tolist() == self.data[: self.left_window_size * 2].tolist()
        )

    def test_n_right_window(self):
        r_i = randint(2, self.data.shape[0] - self.left_window_size)
        l = self.mw.next_window()
        for i in range(r_i):
            r = self.mw.next_right_window()

        self.assertTrue(
            r.tolist()
            == self.data[
                self.left_window_size
                + r_i
                - 1: self.left_window_size
                + r_i
                + self.left_window_size
                - 1
            ].tolist()
        )

    def test_next_left_window(self):
        l = self.mw.next_window()

        next_left_window = self.mw.next_left_window()

        self.assertTrue(
            all(next_left_window == self.data[1: 1 + self.left_window_size])
        )

    def test_n_next_left_windows(self):
        l = self.mw.next_window()
        r_i = randint(2, self.data.shape[0] - self.left_window_size)

        for i in range(r_i):
            next_left_window = self.mw.next_left_window()

        self.assertTrue(
            all(next_left_window ==
                self.data[r_i: r_i + self.left_window_size])
        )

    def test_next_window(self):
        l = self.mw.next_window()
        l2 = self.mw.next_window()

        self.assertTrue(
            l.tolist() + l2.tolist()
            == self.data[: self.stride + self.left_window_size].tolist()
        )

    def test_next_window_with_next_right_window(self):
        l = self.mw.next_window()
        l2 = self.mw.next_window()

        n_r = self.mw.next_right_window()

        self.assertTrue(
            n_r.tolist()
            == self.data[
                self.stride
                + 1
                + self.left_window_size
                - 1: self.stride
                + 1
                + 2 * self.left_window_size
                - 1
            ].tolist()
        )

    def test_next_window_with_next_left_window(self):
        l = self.mw.next_window()
        l2 = self.mw.next_window()

        n_l = self.mw.next_left_window()
        self.assertTrue(
            n_l.tolist()
            == self.data[
                self.stride + 1: self.stride + 1 + self.left_window_size
            ].tolist()
        )

    def test_next_window_with_next_left_and_right_window(self):
        l = self.mw.next_window()
        l2 = self.mw.next_window()

        n_l = self.mw.next_left_window()
        n_r = self.mw.next_right_window()

        self.assertTrue(
            n_l.tolist()
            == self.data[
                self.stride + 1: self.stride + 1 + self.left_window_size
            ].tolist()
        )
        self.assertTrue(
            n_r.tolist()
            == self.data[
                self.stride
                + 1
                + self.left_window_size
                - 1: self.stride
                + 1
                + 2 * self.left_window_size
                - 1
            ].tolist()
        )

    def test_next_window_with_n_left_and_right_window(self):
        self.mw.next_window()
        r_l = randint(2, int((self.data.shape[0] - self.left_window_size) / 4))
        r_r = randint(2, int((self.data.shape[0] - self.left_window_size) / 4))

        for i in range(r_l):
            n_l = self.mw.next_left_window()
        for i in range(r_r):
            n_r = self.mw.next_right_window()

        self.assertTrue(
            n_l.tolist()
            == self.data[
                self.stride + r_l + 1: self.stride + r_l + 1 + self.left_window_size
            ].tolist()
        )

        self.assertTrue(
            n_r.tolist()
            == self.data[
                self.stride
                + r_r
                + self.left_window_size: self.stride
                + r_r
                + int(2 * self.left_window_size)
            ].tolist()
        )

    def setUp(self):
        mkdir("test_files")
        self.data = np.tile([1, 2, 3, 4, 3, 2], 5)
        self.left_window_size = 5
        self.stride = self.left_window_size
        self.max_steps = self.data.shape[0] - self.left_window_size

        self.mw = MissionWindowing(
            self.data, self.left_window_size, self.max_steps, self.stride
        )

    def tearDown(self):
        rmdir("test_files")


class TestCorrealtionAlgorthmusV2(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

    def compare_dicts(self, real_d: dict, d: dict):
        for k, v in real_d.items():
            self.assertTrue(k in d.keys())
            if len(v) == 0:
                self.assertTrue(len(d[k]) == 0)
            else:
                self.assertTrue(len(v) == len(d[k]))
                self.assertTrue(
                    np.all(np.isclose(np.array(v), np.array(d[k]))))

    def test_correlation(self):
        real_c = {
            "0_0": [],
            "0_1": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "0_2": [],
            "1_0": [],
            "1_1": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "1_2": [],
            "2_0": [],
            "2_1": [1, 1, 1, 1],
            "2_2": [],
            "3_0": [],
        }

        c = self.correlation.calc(self.data)

        self.compare_dicts(real_c, c)

    def setUp(self):
        mkdir("test_files")
        self.data = np.tile([1, 2, 3, 4, 3, 2], 4)
        self.left_window_size = 5
        self.stride = self.left_window_size
        self.max_steps = self.data.shape[0] - self.left_window_size

        self.correlation = MissionCorrelationV2(
            self.left_window_size, self.max_steps, self.stride, 0.98
        )

    def tearDown(self):
        rmdir("test_files")


if __name__ == "__main__":
    unittest.main()
