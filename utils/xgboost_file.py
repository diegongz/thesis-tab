import sys
import os

#Let's extract the project path 

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__))

# Get the project directory
project_path = os.path.dirname(current_folder)

print(project_path)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files

from utils import data
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from skopt import BayesSearchCV




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

def xgboost_bayesian(params, X_train, y_train, n_labels):
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class = n_labels,
        seed = 11,
        n_jobs=-1
    )
    
    # Initialize BayesSearchCV
    bayes_search = BayesSearchCV(
        estimator = model,
        search_spaces = params,
        n_iter = 60,   # Number of parameter settings that are sampled
        cv = 5,        # Cross-validation splitting strategy
        n_jobs = -1,   # Use all available cores
        random_state = 11,
        verbose= 2,  # Controls the verbosity: 2 given that we want to monitor how is running
        scoring='balanced_accuracy'
    )

    # Perform the search
    bayes_search.fit(X_train, y_train)

    # Get the best model
    best_model = bayes_search.best_estimator_ #Here the best model is saved
    best_params = bayes_search.best_params_ #here we have the best parameters
    cv_results = bayes_search.cv_results_ #here we have the results of the cross validation
    best_score = bayes_search.best_score_

    return best_model, best_params, cv_results, best_score

def evaluate_xgboost_bayesian(model, X_train, y_train, X_test, y_test):
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Now I will get the metrics
    # Generate confusion matrix
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    #Now I will save the results in a dictionary
    metrics = {
        'balanced_accuracy': balanced_accuracy,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

    return metrics


def hyperparameters_xgboost(params, X_train, y_train, train_indices, val_indices, n_labels, device_name):

    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class = n_labels,
        seed = 11,
        device= device_name,
        early_stopping_rounds=10,
        **params
    )
    
    if device_name == "cuda":
        import cupy as cp
        # Fit the model
        model.fit(cp.array(X_train)[train_indices], cp.array(y_train)[train_indices], eval_set=[(X_train[val_indices], y_train[val_indices])], verbose=True)

    else:
        model.fit(X_train[train_indices], y_train[train_indices], eval_set=[(X_train[val_indices], y_train[val_indices])], verbose=True)

    metrics = {}

    # Predict and evaluate accuracy
    y_pred_in_val = model.predict(X_train[val_indices])

    balanced_accuracy = balanced_accuracy_score(y_train[val_indices], y_pred_in_val)
    accuracy = accuracy_score(y_train[val_indices], y_pred_in_val)
    precision = precision_score(y_train[val_indices], y_pred_in_val, average='weighted')
    recall = recall_score(y_train[val_indices], y_pred_in_val, average='weighted')
    final_n_estimators = model.best_iteration + 1


    metrics = {
        'balanced_accuracy': balanced_accuracy,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'n_estimators': final_n_estimators,
    }
    
    return metrics

def general_row_result(config_num, fold_num, metrics, params):
    config_result = []
    
    config_result.append(config_num)
    config_result.append(fold_num)

    _, metrics_values = zip(*metrics.items())
    metrics_values = list(metrics_values)
    config_result.extend(metrics_values)

    _, params_values = zip(*params.items())
    params_values = list(params_values)
    config_result.extend(params_values)

    return config_result


def run_xgboost(task_id, sample_size):
    
    #define the parameter to do the search
    #'subsample': [0.8, 0.9, 1],
    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2]
    }

    X_train, X_test, y_train, y_test, train_indices, _, _, n_labels, _, _ = data.import_data(task_id, sample_size)

    X_train = X_train[train_indices]
    y_train = y_train[train_indices]


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
    cv_results = grid_search.cv_results_ #here we have the results of the cross validation


    # Evaluate the best model on the test set
    y_pred = best_model.predict(X_test)

    #Now I will get the metrics
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred) #[[TN FP] [FN TP]]
    accuracy = accuracy_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    #f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    #Now I will save the results in a dictionary
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        #'roc_auc': roc_auc
    }

    
    return best_params, metrics, cv_results
    



