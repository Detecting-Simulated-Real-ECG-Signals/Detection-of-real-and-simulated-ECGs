'''
Module contains the custom torch dataset 
'''

from typing import Tuple

import torch


class AllSimulatedMissionsDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        missions_data,
        num_memory_build: int,
        num_predictions: int,
        signal_len: int,
    ):
        self.data = missions_data

        self.num_memory_build: int = num_memory_build
        self.num_predictions: int = num_predictions
        self.signal_len: int = signal_len

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        mission_idx = int(
            idx / (self.signal_len - (self.num_memory_build + self.num_predictions))
        )
        s_value = idx % (
            self.signal_len - (self.num_memory_build + self.num_predictions)
        )

        m_d, _ = self.data.get_by_index(self.data.indices[mission_idx])
        s = m_d[s_value:]

        return (
            s[: self.num_memory_build],
            s[self.num_memory_build: self.num_memory_build + self.num_predictions],
        )

    def __len__(self) -> int:
        return len(self.data.indices) * (
            self.signal_len - (self.num_memory_build + self.num_predictions)
        )
