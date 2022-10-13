"""
## define dataset class
# load data set from csv
"""


import sys
import numpy as np
import torch
from torch.utils.data import Dataset


## create Pytorch Dataset ##  Dataset stores the samples and their corresponding labels
class Sequence_Dataset(Dataset):

    def __init__(self, sequences):
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence, label = self.sequences[idx]
        return dict(
            sequence=torch.Tensor(sequence.to_numpy()),
            label=torch.tensor(label.to_numpy()).long().squeeze()
        )

