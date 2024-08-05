import sys
import os

#Let's extract the project path 

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__))

# Get the project directory
project_path = os.path.dirname(current_folder)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files

from utils import data
import numpy as np
import torch
import torch.nn as nn
from utils import training, callback, evaluating, attention, data, plots, fast_model
from sklearn import datasets, model_selection
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets, model_selection
import csv
import xgboost as xgb



'''
We need some values to apply the grid search in the xgboost

'max_depth': [3, 4, 5],
'n_estimators': [100, 200, 300], #number of trees that will be trained
'learning_rate': [0.1, 0.01, 0.05],
'gamma': [0, 0.25, 1.0],
'subsample': [0.8, 1],
'reg_lambda': [0, 1.0, 10.0],
'colsample_bytree': [0.6, 0.8, 0.5]

'''




def xgboost(task_id, sample_size):
    
    #define the parameter to do the search
    #'subsample': [0.8, 0.9, 1],
    param_grid = {
    'max_depth': [3, 4, 5],
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 1.0],
    'reg_lambda': [0, 1.0, 10.0],
    'colsample_bytree': [0.6, 0.8, 0.5]
    }

    X_train, X_test, y_train, y_test, _, _, _, n_labels, _, _ = data.import_data(task_id, sample_size)

    clf_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class = n_labels, seed = 11, device= 'cuda')

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=clf_xgb,
                            param_grid=param_grid,
                            scoring='balanced_accuracy',  # Adjust scoring metric as needed
                            cv=5,  # 5-fold cross-validation
                            n_jobs=-1, # Use all cores
                            verbose= 2)  #verbose help to print progress
    
    # Fit the grid search to the training set
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_ #Here the best model is saved
    best_params = grid_search.best_params_ #here we have the best parameters

    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)

    #Now I will get the metrics
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    #roc_auc = roc_auc_score(y_test, y_pred, average='weighted')

    #Now I will save the results in a dictionary
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        #'roc_auc': roc_auc
    }

    # Add these metrics to best_params
    best_params.update(metrics)


    return best_params
    



