import sys
sys.path.append('/home/diego/Git/thesis-tabtrans')
project_path = '/home/diego/Git/thesis-tabtrans'
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


FINAL DATASETS
12 mfeat-factors 217 pass
20 mfeat-pixel 241 #have probelms this dataset check this one
9964 semeion 257 pass
233092 arrhythmia 280
3485 scene 300
9976 madelon 501
3481 isolet 618 (TO MUCH INSTANCES) 


---------Datasets with id not task id----------
1484 lsvt
1564 dbworld-subjects
'''

#tasks = [233090]
#tasks = [1484, 12, 9964, 233092, 3485, 9976] #1484,1564, 12, 9964, 233092, 3485, 9976


#I need to finish the experiments 
tasks = [233092, 3485, 9976] 

#sample_size = [100,80,60,40,20] #100,80,60,40,20
project_path = '/home/diego/Git/thesis-tabtrans'

for task_id in tasks:
    if task_id == 233092:
        sample_size = [60,40,20] #because i have already done the ones before this
        fast_model.train_xgboost(task_id, sample_size, project_path)

    else:
        sample_size = [100,80,60,40,20] #100,80,60,40,20
        fast_model.train_xgboost(task_id, sample_size, project_path)
