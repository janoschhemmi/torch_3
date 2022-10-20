import torch
import pytorch_lightning
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision
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
from pytorch_lightning.callbacks import TQDMProgressBar
from torchmetrics.functional import accuracy
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning import loggers as pl_loggers



class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, dropout, batch_norm_trigger):

        """
        ## general input of an LSTM: [batch_size, seq_len, input_size]

        :param num_classes:  Number of classes to target
        :param input_size:   Number of features per time step
        :param hidden_size:  Size of Hidden and Cell State
        :param num_layers:   Number of LSTM Layers stacked over each other
        :param seq_length:   Number of time steps per sample
        :param dropout:      ratio of dropout at each LSTM Layer
        """

        super(LSTM1, self).__init__()
        self.num_classes = num_classes  # number of classes
        self.num_layers = num_layers    # number of lstm stacked
        self.input_size = input_size    # Number of features per time step
        self.hidden_size = hidden_size  # size of hidden state
        self.seq_length = seq_length    # length of sequence
        self.dropout  = dropout         # dropout rate

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)  # lstm
        self.fc = nn.Linear(hidden_size, num_classes)  # fully connected out
        self.fc_2 = nn.Linear(50, num_classes)  # fully connected out

        self.relu = nn.ReLU()
        self.batch = nn.BatchNorm1d(hidden_size)
        self.batch_norm_trigger = batch_norm_trigger


    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        #print('x in 1: {}'.format(x.shape))

        ## transform input x
        x = torch.transpose(x,1,2)
        #print('x in 1: {}'.format(x.shape))
        x = torch.tensor(x, dtype = torch.float32)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0.detach(), c_0.detach()) )  # lstm with input, hidden, and internal state

        if (self.num_layers == 1 ):
            hn = hn[-1,:,:]
            #print("output: ", hn.shape)

        ## just take hn of last layer
        if (self.num_layers != 1 ):
            hn = hn[-1,:,:]
            #print("output: ", hn.shape)

        """
        if self.batch_norm_trigger:
            out = (self.fc(self.batch(self.relu(hn))))  # Final Output
        else:
            out = (self.fc((self.relu(hn))))  # Final Output

        return self.fc_2(out)"""

        return (self.fc((self.relu(hn))))


## Define TRaining and Validation steps
class Disturbance_Predictor_model_lstm(pl.LightningModule):
    def __init__(self, n_classes: int,  input_size=int, seq_length=25, size_of_hidden_state=50 ,num_layers=2 ,
                 learning_rate = 0.001, drop_out = 0.2, batch_norm_trigger = False, batch_size = 100):
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







