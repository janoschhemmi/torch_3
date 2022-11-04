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


