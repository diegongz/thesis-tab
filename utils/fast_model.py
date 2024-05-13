import os
import sys
import numpy as np
import torch
import torch.nn as nn
from utils import training, callback, evaluating, attention, data, plots 
from sklearn import datasets, model_selection
import skorch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import openml
from sklearn import datasets, model_selection
from skorch.callbacks import Checkpoint, EarlyStopping, LoadInitState, EpochScoring, Checkpoint, TrainEndCheckpoint

def create_parameters(task_id, Layers, Heads, Emedding_dim):
    
    parameters = {
    "task_id" : task_id,
    "Layers": Layers,
    "Heads": Heads,
    "Emedding_dim": Emedding_dim
    }

    return parameters

'''
Input a project_path and the name of the new folder you want to create.

If the folder doesn't exist, it will create a new folder in the project_path.
If the folder already exists, it will do nothing.
'''
def new_folder(project_path, new_folder_name):
    # Create the full path for the new folder
    new_folder_path = os.path.join(project_path, new_folder_name)

    # Check if the new folder exists
    if not os.path.exists(new_folder_path):
        # If it doesn't exist, create the new folder
        os.makedirs(new_folder_path)