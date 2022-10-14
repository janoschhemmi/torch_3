import argparse

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import Dataset, DataLoader
import torchvision
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from torchmetrics.functional import accuracy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import multiprocessing
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from matplotlib.ticker import MaxNLocator


from data import load_data_set
from data import SequenceDataModule, TimeSeriesDataset
from data import data_labeler

from essentials import other_functions
from essentials import check_path



SequenceDataModule(train_sequences=)


## Globals #######
plt.switch_backend('agg')

## X train path
x_train_data_path = r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\06_df_x_12_250smps.csv"
y_train_data_path = r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\06_df_y_12_250smps.csv"

## path to safe model
save_model_path = r"P:\workspace\jan\fire_detection\dl\models_store\07_LSTM"
check_path(save_model_path)

## model params
N_EPOCHS = 1
BATCH_SIZE = 10

## training logger path
logger_path = "P:/workspace/jan/fire_detection/dl/models_store/07_LSTM/tl_logger/"
check_path(logger_path)
logger_name = "Disturbance_predictor_9"

## ---------------------------------------------------------------------------------------------------------------------
## Main

if __name__ ==  '__main__':

    ## read data
    X_train = pd.read_csv(x_train_data_path, sep=';')
    y_train = pd.read_csv(y_train_data_path, sep=';')
    print(X_train.shape, y_train.shape)

    # data label
    y_train = data_labeler(y_train)

    ## data set




    # reshape
    X_train = (X_train.set_index(['id', 'index', 'instances_rep'])
               .rename_axis(['step'], axis=1)
               .stack()
               .unstack('index')
               .reset_index())
    FEATURE_COLUMNS = X_train.columns.tolist()[3:]

    ## group per disturbance instance for individual sequence
    sequences = []
    for instances, group in X_train.groupby("instances_rep"):
        print(id)
        sequence_features = group[FEATURE_COLUMNS]
        sequence_features = sequence_features.transpose()

        label = y_train[y_train.instances_rep == instances].label
        print(label)
        sequences.append((sequence_features, label))
    jj = sequences[0]
    jj[0]