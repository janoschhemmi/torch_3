"""
## define dataset class
# load data set from csv
"""


import sys
import numpy as np
import torch
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import Dataset
import pandas as pd


###
def index_scaler(Xs):

    colnames_keep = Xs.columns
    index_list = list(Xs['index'].unique())

    Xs = (Xs.set_index(['id', 'index', 'instances_rep'])
                 .rename_axis(['step'], axis=1)
                 .stack()
                 .unstack('index')
                 .reset_index())

    ## normalize per row
    minmax_scale = MinMaxScaler(feature_range=(0, 1)).fit(Xs[Xs.columns.intersection(index_list)])
    X_minmax = minmax_scale.transform(Xs[Xs.columns.intersection(index_list)])

    ## concat
    x_ = pd.concat([Xs[Xs.columns.intersection(['id',  'instances_rep', 'step'])],pd.DataFrame(X_minmax)], axis = 1)
    zz = x_.set_index(['id', 'instances_rep','step']).stack().unstack(2).sort_index(axis=1, ascending=False).reset_index()
    zz = zz.rename(columns = {"level_2":"index"})
    zzz = zz[colnames_keep]
    zzz['index'] = zzz['index'].replace({0:index_list[0],1:index_list[1],2:index_list[2],
                                         3:index_list[3],4:index_list[4],5:index_list[5],
                                         6:index_list[6],7:index_list[7]})

    return zzz



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
            sequence= sequence,
            label= label.type(torch.LongTensor)
            #label=label.long().squeeze()
        )

## create data set

"""X_train

for instances, group in X_train.groupby("instances_rep"):
    print(id)
    sequence_features = group[FEATURE_COLUMNS]
    sequence_features = sequence_features.transpose()

    label = y_train[y_train.instances_rep == instances].label
    print(label)
    sequences.append((sequence_features, label))"""



class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, Xs, ys, n_steps):
        super(torch.utils.data.Dataset,self).__init__()

        ## convert to tensor
        X = [torch.tensor(group[group.columns[-n_steps:]].values) for instance,group in Xs.groupby("instances_rep")]
        y = torch.tensor(ys['label']).type(torch.LongTensor)

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


