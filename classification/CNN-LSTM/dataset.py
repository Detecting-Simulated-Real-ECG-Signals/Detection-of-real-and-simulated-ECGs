'''
Module contains the custom torch dataset 
'''

from random import choices, random
from typing import Dict, List, Tuple

import torch
from defibrilator_preprocessing.signal_scaling import threshold_scale


def horizontal_flip(input):
    """
    flip scaled signal randomly horizontally
    """
    return input * choices([-1, 1])[0]  # use choice because -1 is horizontal flip while 1 keeps data as it is


def random_offset(input: torch.Tensor):
    """
    move scaled signal randomly up and down

    **Expects normalized data!**
    """
    shift_margin = 1 - input.max() + 1 - input.min()

    return input - input.min() + (random() * shift_margin)


class LocalDataSetWorker(torch.utils.data.Dataset):
    def __init__(
        self,
        loader_method,
        indices: List[int],
        map_labels: Dict[str, int],
        test: bool = False,
    ):
        self.loader_method = loader_method
        self.indices = indices
        self.map_labels = map_labels
        self.test = test

    # Thresholds my vary for you. These thresholds where determined for my dataset.
    def __getitem__(self, idx, lower_threshold: float = -1500, upper_threshold: float = 1600) -> Tuple[torch.Tensor, torch.Tensor]:
        i = self.indices[idx]
        d, l = self.loader_method(i)

        label = torch.zeros((2))
        label[self.map_labels[l]] = 1

        data = torch.tensor(threshold_scale(
            d, lower_threshold, upper_threshold)).float()

        if not self.test:
            return random_offset(horizontal_flip(data)), label.float()
        else:
            return data, label.float()

    def __len__(self) -> int:
        return len(self.indices)
