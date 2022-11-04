import torch
import pytorch_lightning as pl
from .load_data_set import *
from torch.utils.data import Dataset, DataLoader




## DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
class SequenceDataModule(pl.LightningDataModule):
    def __init__(self, train_sequences, test_sequences, batch_size, shuffle):
        super().__init__()
        self.train_sequences = train_sequences
        self.test_sequences = test_sequences
        self.batch_size = batch_size
        self.shuffle = shuffle

    def setup(self, stage=None):
        self.train_dataset = Sequence_Dataset(self.train_sequences)
        self.test_dataset  = Sequence_Dataset(self.test_sequences)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            #num_workers=cpu_count()
            num_workers = 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

