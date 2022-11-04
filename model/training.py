import argparse
import random
import time

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



## Define TRaining and Validation steps
class Disturbance_Predictor_model_lstm(pl.LightningModule):
    def __init__(self, n_classes: int,  input_size=int, seq_length=25, size_of_hidden_state=50 ,num_layers=2 ,
                 learning_rate = 0.001, drop_out = 0.2, batch_norm_trigger = False,
                batch_size = 100, hidden_size = 100):
        super().__init__()


        self.model = LSTM1(n_classes,input_size,size_of_hidden_state,num_layers,seq_length, drop_out, batch_norm_trigger)
        self.criterion = nn.CrossEntropyLoss()

        self.learning_rate = learning_rate
        self.learning_rate_hp = learning_rate

        self.batch_size = batch_size
        self.batch_size_hp = batch_size

        self.step_accuracy = 0
        self.step_accuracy_hp = 0

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams, {"hp/learning_ratee": self.learning_rate_hp, "hp/batch_size": self.batch_size_hp,
                                                   "hp/step_accuracy": self.step_accuracy_hp})


    def training_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]
        #labels = labels.squeeze()

        loss, outputs = self.forward(sequences, labels)
        #print("output: ", outputs)
        predictions   = torch.argmax(outputs, dim=1)
        #print("prediction:", predictions)
        #print("prediction:", labels)

        step_accuracy = accuracy(predictions, labels)
        self.step_accuarcy = step_accuracy
        self.step_accuarcy.hp = step_accuracy

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        print("here")

        sequences = batch["sequence"]
        labels = batch["label"]

        #print(sequences)
        loss, outputs = self.forward(sequences, labels)
        print("output ", outputs[1:5])
        print("label ", labels[1:5])
        predictions   = torch.argmax(outputs, dim=1)
        print("prediction ", predictions[1:5])


        step_accuracy = accuracy(predictions.long(), labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)
        self.log("hp/learning_rate", self.learning_rate)
        self.log("hp/batch_size", self.batch_size)
        """self.logger.log_hyperparams(self.hparams,
                                    {"hp/learning_ratee": self.learning_rate_hp, "hp/batch_size": self.batch_size_hp,
                                     "hp/step_accuracy": self.step_accuracy_hp})"""

        return {"loss": loss, "accuracy": step_accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr= self.learning_rate)

    def training_epoch_end(self, outputs):
        #  the function is called after every epoch is completed

        # calculating average loss
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()

        self.logger.experiment.add_scalar("Val Loss",
                                            avg_loss,
                                            self.current_epoch)

        self.logger.experiment.add_scalar("Val Acc",
                                          avg_acc,
                                          self.current_epoch)






