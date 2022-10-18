import torch
import pytorch_lightning

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




class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, dropout):

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
        self.relu = nn.ReLU()


    def forward(self, x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))  # internal state
        #print('x in 1: {}'.format(x.shape))

        ## transform input x
        x = torch.transpose(x,1,2)
        #print('x in 1: {}'.format(x.shape))
        x = torch.tensor(x, dtype = torch.float32)

        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0) )  # lstm with input, hidden, and internal state

        ## just take hn of last layer
        if (self.num_layers != 1 ):
            hn = hn[-1,:,:]
        return(self.fc(self.relu(hn)))  # Final Output



## Define TRaining and Validation steps
class Disturbance_Predictor_model_lstm(pl.LightningModule):
    def __init__(self, n_classes: int,  input_size=int, seq_length=25, size_of_hidden_state=100 ,num_layers=2 ):
        super().__init__()


        self.model = LSTM1(n_classes,input_size,size_of_hidden_state,num_layers,seq_length,dropout = 0.5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        output = self.model(x)
        loss = 0
        if labels is not None:
            loss = self.criterion(output, labels)
        return loss, output

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

        self.log("train_loss", loss, prog_bar=True, logger=True)
        self.log("train_accuracy", step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}

    def validation_step(self, batch, batch_idx):
        sequences = batch["sequence"]
        labels = batch["label"]

        loss, outputs = self.forward(sequences, labels)
        predictions   = torch.argmax(outputs, dim=1)
        step_accuracy = accuracy(predictions.long(), labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_accuracy", step_accuracy, prog_bar=True, logger=True)

        return {"loss": loss, "accuracy": step_accuracy}

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=0.0005)


