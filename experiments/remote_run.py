import sys
sys.path.append('/home/diego_ngz/Git/thesis-tabtrans')
project_path = '/home/diego_ngz/Git/thesis-tabtrans'
from utils import data
import os
import numpy as np
import torch
import torch.nn as nn
from utils import training, callback, evaluating, attention, data, plots, fast_model
from sklearn import datasets, model_selection
import skorch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import openml
from sklearn import datasets, model_selection
from skorch.callbacks import Checkpoint, EarlyStopping, LoadInitState, EpochScoring, Checkpoint, TrainEndCheckpoint
import csv

#task_id = 233090 #anneal
#task_id = 233093 #mfeat
#task_id = 233092 #arrhythmia
#task_id = 233108 #cnae-9
#task_id = 233118 #fashion MNIST
''' 
233118 fashion-mnist 784 features
233133 falbert 801 features
233108 cnae-9 857 features
233121 Devnagari-Script 1025 features
233131 christine 1637 features 
233132 dilbert 2001 features

'''

tasks = [233090]

n_layers_lst = [2] #2, 3, 4, 5
n_heads_lst = [4] #4, 8, 16, 32
embed_dim = [128] #The embedding size is set one by one to avoid the out of memory error {128, 256}
batch_size = 32 # 32, 64, 128, 256, 512, 1024
epochs = 2
sample_size = [40, 20]
project_path = '/home/diego_ngz/Git/thesis-tabtrans'

for task_id in tasks:
    fast_model.train_model(task_id, sample_size, n_layers_lst, n_heads_lst, embed_dim, batch_size, epochs, project_path)

