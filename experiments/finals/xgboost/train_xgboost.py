import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of this file

# Get the project directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_folder))) #parent folder of current_folder

print(project_path)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files


from utils import data
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb


'''
FINAL DATASETS
1484 lsvt
31 credit 
12 mfeat-factors 217 pass
20 mfeat-pixel 241 #have probelms this dataset check this one
9964 semeion 257 pass
233092 arrhythmia 280
3485 scene 300
9976 madelon 501
3481 isolet 618 (TO MUCH INSTANCES) 

'''


df_id = 1484
sample_size = [100]

name_df = data.get_dataset_name(df_id)

path_of_datset = f'{project_path}/Final_models_4/{name_df}' #The path can be changed

path_of_xgboost = f'{path_of_datset}/xgboost'

os.makedirs(path_of_datset, exist_ok=True)
os.makedirs(path_of_xgboost, exist_ok=True)

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1],
    'reg_lambda': [1, 3, 5],
    'n_estimators': [100,300,500],
    'gamma': [0, 0.1, 1],
    #'reg_alpha': [0, 0.1, 0.5] #this is for L1 regularization that can be added but the original objective function dont have L! regularization
    }


for sample in sample_size:
    if sample == 100:

        #import data for the task
        X_train, X_test, y_train, y_test, _, _, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

        clf_xgb = xgb.XGBClassifier(objective='multi:softmax', num_class = n_labels, seed = 11, device= 'cuda')


