import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of this file

# Get the project directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_folder))) #parent folder of current_folder

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files


from utils import data, xgboost_file
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from itertools import product
import time



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
device = "cpu"
seed = 11

name_df = data.get_dataset_name(df_id)

path_of_datset = f'{project_path}/Final_models_4/{name_df}' #The path can be changed

path_of_xgboost = f'{path_of_datset}/xgboost/hyperparameter_selection'

os.makedirs(path_of_datset, exist_ok=True)
os.makedirs(path_of_xgboost, exist_ok=True)

param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.5, 0.8, 1],
    'colsample_bytree': [0.5, 0.8, 1],
    'reg_lambda': [1, 3, 5],
    'n_estimators': [100,300,500],
    'gamma': [0, 0.1, 1]
    }

start_time = time.time()
for sample in sample_size:
    if sample == 100:
        #create the folder of the sample size
        size_path = f'{path_of_xgboost}/{sample}'
        os.makedirs(size_path, exist_ok=True)


        #import data for the task
        X_train, X_test, y_train, y_test, _, _, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

        keys, values = zip(*param_grid.items()) #extract the keys and values of the dictionary

        names_of_parameters = list(keys)

        all_experiments_results = []
        all_experiments_results_names = ["config_num", "fold_num","balanced_accuracy", "accuracy", "precision", "recall", "n_estimators" ,"confusion_matrix"]
        all_experiments_results_names.extend(names_of_parameters)
        
    

        config_num = 1
        
        #Variables to compare the results of the different configurations
        best_params = None
        best_balanced_accuracy = 0

        for combination in product(*values): #This selects an specific configuration for the xgboost
            params = dict(zip(keys, combination))
            print("---------------------------------------------------------------------")
            print(f"Evaluating parameters: {params}")
            
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = seed)
            
            fold_num = 1
            folds_results_balanced_accuracies = [] #The result of the folds will be stored here to later average them to find the mean of the metric 
            fold_results_n_estimators = []

            for train_indices,val_indices in skf.split(X_train, y_train):
                
                print("-------------------------------------------------------------------")
                print(f"Fold Number: {fold_num}")
                metrics = xgboost_file.hyperparameters_xgboost(params, X_train, y_train, train_indices, val_indices, n_labels, device)
                

                config_result = xgboost_file.general_row_result(config_num, fold_num, metrics, params)
                all_experiments_results.append(config_result)

                fold_balanced_accuracy = metrics['balanced_accuracy']
                folds_results_balanced_accuracies.append(fold_balanced_accuracy)

                fold_n_estimators = metrics['n_estimators']
                fold_results_n_estimators.append(fold_n_estimators)
                
                
                fold_num += 1

            mean_balanced_accuracy = sum(folds_results_balanced_accuracies)/len(folds_results_balanced_accuracies)
            mean_n_estimators = sum(fold_results_n_estimators)/len(fold_results_n_estimators)


            if mean_balanced_accuracy > best_balanced_accuracy:
                best_balanced_accuracy = mean_balanced_accuracy
                best_params = params

            config_num += 1
        
        #Save the results of the experiments
        best_params["balanced_accuracy_mean"] = best_balanced_accuracy
        best_params["n_estimators_mean"] = mean_n_estimators

        # Convert dictionary to DataFrame
        results = pd.DataFrame([best_params])
        results_path = os.path.join(size_path, 'results.csv')
        results.to_csv(results_path, index=False)

        #genral results df
        general_results_df = pd.DataFrame(all_experiments_results, columns = all_experiments_results_names)
        general_results_path = os.path.join(size_path, 'general_results.csv')
        general_results_df.to_csv(general_results_path, index=False)



# End the timer
end_time = time.time()
# Calculate and print the elapsed time
elapsed_time = end_time - start_time
print(f"Training took {elapsed_time:.2f} seconds.")














