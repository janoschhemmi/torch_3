import argparse
import random

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
from sklearn.preprocessing import Normalizer

import multiprocessing
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib
from matplotlib.ticker import MaxNLocator
import random


from data import load_data_set
from data import SequenceDataModule
from data import data_labeler
from data import TimeSeriesDataset
from data import index_scaler

from essentials import other_functions
from essentials import check_path

from model import Disturbance_Predictor_model_lstm


## Globals #######
plt.switch_backend('agg')
random.seed(102)

## X train path
x_train_data_path = r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\06_df_x_12_250smps.csv"
y_train_data_path = r"P:\workspace\jan\fire_detection\dl\prepocessed_ref_tables\06_df_y_12_250smps.csv"

## path to safe model
save_model_path = r"P:\workspace\jan\fire_detection\dl\models_store\07_LSTM"
check_path(save_model_path)

## model params
N_EPOCHS = 1
BATCH_SIZE = 10

n_features = 8

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

    X_train_2 = index_scaler(X_train)

    # data label
    y_train   = data_labeler(y_train)

    ## data set
    dset = TimeSeriesDataset(X_train, y_train, 25)

    ## split train test
    train_sequences, test_sequences = train_test_split(dset, test_size=0.2, random_state= 102)
    print("Number of Training Sequences: ", len(train_sequences))
    print("Number of Testing Sequences: ", len(test_sequences))

    ## create Data Module
    Dmod = SequenceDataModule(train_sequences, test_sequences, 32)

    # Model learner
    print("initializing model")
    model = Disturbance_Predictor_model_lstm(
        input_size=n_features ,
        n_classes=len(y_train["label"].unique()))