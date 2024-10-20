import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of this file

# Get the project directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_folder))) #parent folder of current_folder

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files


from utils import data, xgboost_file
import pandas as pd
from itertools import product
import time
from skopt import BayesSearchCV
from skopt.space import Real, Integer
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

'''
PARAMETERS OF THE GRID
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

'''


df_id = 31
sample_size = [100]
seed = 11

name_df = data.get_dataset_name(df_id)

path_of_datset = f'{project_path}/Final_models_4/{name_df}' #The path can be changed
os.makedirs(path_of_datset, exist_ok=True)

path_of_xgboost = f'{path_of_datset}/xgboost'
os.makedirs(path_of_xgboost, exist_ok=True)


# Define parameter space for Bayesian optimization
search_space = {
    'learning_rate': Real(0.01, 0.1, prior='log-uniform'),  # Continuous range, exploring learning rates between 0.01 and 0.1
    'max_depth': Integer(3, 7),                       # Discrete integer values, between 3 and 7                                    # Continuous range for gamma parameter
    'subsample': Real(0.5, 1.0),                           # Continuous range between 0.5 and 1.0
    'colsample_bytree': Real(0.5, 1.0),                    # Continuous range between 0.5 and 1.0
    'reg_lambda': Real(1, 5),                              # Continuous range between 1 and 5 for L2 regularization
    'n_estimators': Integer(100, 500),                     # Discrete range for number of estimators
    'gamma': Real(0, 1)
}

for sample in sample_size:
    if sample == 100:
        path_hyperparameter_selection = f'{path_of_xgboost}/hyperparameter_selection'
        #create the folder of the sample size
        size_path_hyper = f'{path_hyperparameter_selection}/{sample}'
        os.makedirs(size_path_hyper, exist_ok=True)


        #import data for the task
        X_train, X_test, y_train, y_test, _, _, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)
        
        best_model, best_params, cv_results, best_score = xgboost_file.xgboost_bayesian(search_space, X_train, y_train, n_labels)

        results_param_selection = best_params

        # update the best parameters with the best score
        results_param_selection['balanced_accuracy_mean'] = best_score

        df_param = pd.DataFrame(results_param_selection, index=[0])
        df_param.to_csv(f'{size_path_hyper}/results.csv', index=False)

        # Save the cross-validation results
        cv_results = pd.DataFrame(cv_results)
        cv_results.to_csv(f'{size_path_hyper}/cv_results.csv', index=False)

        # Evaluate the best model on the test set
        results = xgboost_file.evaluate_xgboost_bayesian(best_model, X_train, y_train, X_test, y_test)

        results.update(best_params)

        #Save results
        path_of_final_xgboost = f'{path_of_datset}/xgboost/final_xgboost/{sample}'
        os.makedirs(path_of_final_xgboost, exist_ok=True)

        df_results = pd.DataFrame(results, index=[0])
        df_results.to_csv(f'{path_of_final_xgboost}/results.csv', index=False)







