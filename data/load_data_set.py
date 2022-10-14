"""
## define dataset class
# load data set from csv
"""


import sys
import numpy as np
import torch
import sklearn
from torch.utils.data import Dataset

## labeler
def data_labeler(y):
    """
    returns labeled column based on disturbane column
    :param y:
    :return:
    """
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y.disturbance)
    # Adding to y_train df
    y["label"] = encoded_labels
    return y

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

## create data set

X_train

for instances, group in X_train.groupby("instances_rep"):
    print(id)
    sequence_features = group[FEATURE_COLUMNS]
    sequence_features = sequence_features.transpose()

    label = y_train[y_train.instances_rep == instances].label
    print(label)
    sequences.append((sequence_features, label))


tt = X_train[:32]
class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, Xs, ys, Index_list, n_steps):
        super(torch.utils.data.Dataset,self).__init__()
        self.index_list = Index_list


        ## convert to tensor
        X = [torch.tensor(group[group.columns[-n_steps:]].values) for instance,group in Xs.groupby("instances_rep")]

        ## create label for y

        y

        for group in tt.groupby("instances_rep"):
            print(group)


        ttt = tt.groupby("instances_rep")
        sequences = []
        tt[chr(n_steps_list)]

        n_steps = 25



    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        sequence_features = group[FEATURE_COLUMNS]
        sequence_features = sequence_features.transpose()
        return self.X[idx], self.y[idx]
