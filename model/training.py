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


def train_model(model, criterion, optimizer, history,data_module ,batch_size ,scheduler=None,
                num_epochs, save_path='checkpoint', continue_training=False, start_epoch=0):

    # load trained model
    if continue_training:
        with open(BASE_PATH + 'weights/{}_{}.model'.format(save_path, start_epoch - 1), 'rb') as f:
            state = torch.load(f, map_location=DEVICE)
            model.load_state_dict(state)
        with open(BASE_PATH + 'weights/{}_{}.optimizer'.format(save_path, start_epoch - 1), 'rb') as f:
            state = torch.load(f, map_location=DEVICE)
            optimizer.load_state_dict(state)
        with open(BASE_PATH + 'weights/{}_{}.history'.format(save_path, start_epoch - 1), 'rb') as f:
            history = torch.load(f)
        if scheduler:
            with open(BASE_PATH + 'weights/{}_{}.scheduler'.format(save_path, start_epoch - 1), 'rb') as f:
                state = torch.load(f, map_location=DEVICE)
                scheduler.load_state_dict(state)

    num_epochs = 2
    data_module = Data_module
    batch_size = 25

    for epoch in range(start_epoch, num_epochs):
        print("Epoch: ", epoch)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

                all_loader = torch.utils.data.DataLoader(
                    data_module.train_sequences, batch_size=batch_size,
                    shuffle=False
                )

            else:
                model.eval()   # Set model to evaluate mode
                all_loader = torch.utils.data.DataLoader(
                    data_module.test_sequences, batch_size=batch_size,
                    shuffle=False
                )

            running_metrics = {}

            """Iterate over data.
            `dataloaders` is a dict{'train': train_dataloader
                                    'val': validation_dataloader}
            """
            iterator = tqdm(all_loader)
            print(iterator)
            for batch in iterator:
                """
                Batch comes as a dict.
                """
                #for k in batch:
                #    batch[k] = batch[k].to(DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    ## look into what is in a batch and try to parse it to the model
                    outputs = model(batch['src'],
                                    batch['dst'],
                                    batch['src_lengths'],
                                    batch['dst_lengths'])
                    _, preds = outputs.max(dim=2)

                    loss = criterion(outputs.view(-1, len(train_dataset.src_token2id)), batch['dst'].view(-1))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
                        optimizer.step()

                # statistics
                running_metrics.setdefault('loss', 0.0)
                running_metrics['loss'] += loss.item() * batch['src'].size(0)
                for pred, ground_truth in zip(preds, batch['dst']):
                    metrics = get_metrics(pred, ground_truth)       # supposed to return a dictionary of metrics
                    for metric_name in metrics:
                        running_metrics.setdefault(metric_name, 0.0)
                        running_metrics[metric_name] += metrics[metric_name]

            for metric_name in running_metrics:
                multiplier = 1
                average_metric = running_metrics[metric_name] / dataset_sizes[phase]
                history.setdefault(phase, {}).setdefault(metric_name, []).append(average_metric * multiplier)

            print('{} Loss: {:.4f} Rouge: {:.4f}'.format(
                phase, history[phase]['loss'][-1], history[phase]['rouge-l'][-1]))

            # LR scheduler
            if scheduler and phase == 'val':
                scheduler.step(history['val']['loss'][-1])

        # save model and history
        with open(BASE_PATH + 'weights/{}_{}.model'.format(save_path, epoch), 'wb') as f:
            torch.save(model.state_dict(), f)
        with open(BASE_PATH + 'weights/{}_{}.optimizer'.format(save_path, epoch), 'wb') as f:
            torch.save(optimizer.state_dict(), f)
        with open(BASE_PATH + 'weights/{}_{}.history'.format(save_path, epoch), 'wb') as f:
            torch.save(history, f)
        if scheduler:
            with open(BASE_PATH + 'weights/{}_{}.scheduler'.format(save_path, epoch), 'wb') as f:
                torch.save(scheduler.state_dict(), f)


        time_elapsed = time.time() - since
        history.setdefault('times', []).append(time_elapsed)     # save times per-epoch
        print('Epoch {} complete in {:.0f}m {:.0f}s'.format(epoch,
            time_elapsed // 60, time_elapsed % 60))
        print()


