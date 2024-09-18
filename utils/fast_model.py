import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__))

# Get the project directory
project_path = os.path.dirname(current_folder)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files


import numpy as np
import torch
import torch.nn as nn
from utils import training, callback, evaluating, attention, data, plots, xgboost_file
from sklearn import datasets, model_selection
import skorch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import openml
from sklearn import datasets, model_selection
from skorch.callbacks import Checkpoint, EarlyStopping, LoadInitState, EpochScoring, Checkpoint, TrainEndCheckpoint
import csv
import time

def create_parameters(task_id, Layers, Heads, Emedding_dim, batch_size, epochs):
    
    parameters = {
    "task_id" : task_id,
    "Layers": Layers,
    "Heads": Heads,
    "Emedding_dim": Emedding_dim,
    "batch_size": batch_size,
    "epochs": epochs,
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

def export_to_csv(results_table, columns_names, folder_path):
    # Creating the file path
    file_path = folder_path + "/results.csv"

    # Writing to the CSV file
    with open(file_path, 'w', newline='') as csvfile:
        # Creating a CSV writer object
        csv_writer = csv.writer(csvfile)

        # Writing the column names
        csv_writer.writerow(columns_names)

        # Writing the data rows
        csv_writer.writerows(results_table)


'''
This function trains the model with given parameters

Parameters:
sample_size: list of the possible size percentage you want from the original dataset
n_layers_lst: list of integers, the number of layers in the transformer
n_heads_lst: list of integers, the number of heads in the transformer
embed_dim: integer, the embedding size of the transformer (Integer so you can control the GPU out of memory error)
batch_size: integer, the batch size of the training
epochs: integer, the number of epochs to train the model
'''

def train_model(task_id, sample_size, n_layers_lst, n_heads_lst, embed_dim, batch_size, epochs, project_path):
    for percentage in sample_size:    
        #From Id get the data

        X_train, X_test, y_train, y_test, train_indices, val_indices, _, n_labels, n_numerical, n_categories = data.import_data(task_id, percentage)

        #get the dataset_name
        if task_id in [1484,1564]: #two selected datasets with no task id, just id
            dataset = openml.datasets.get_dataset(task_id)
            dataset_name = dataset.name
        else:
            dataset_name = data.get_dataset_name(task_id)

        #Find if I have multiclass in the y's
        if len(np.unique(y_train)) > 2:
            multiclass_val = True
        
        else:
            multiclass_val = False

        print(np.unique(y_train))
    

        #create the folder to save the dataset experiments if it doesn't exist
        new_folder(project_path, "Final_models_2")
        path_of_data_models = os.path.join(project_path, "Final_models_2") #path of the folder data_models

        #create the folder for specific dataset name
        new_folder(path_of_data_models, f"{dataset_name}") #_{embed_dim}") #
        path_of_dataset = os.path.join(path_of_data_models, f"{dataset_name}") #path of the folder dataset 

        #create the tabular-transformer folder
        new_folder(path_of_dataset, "tabtrans") 
        path_of_tabtrans = os.path.join(path_of_dataset, "tabtrans") #path of the folder dataset

        #create folder of hyperparameter selection
        new_folder(path_of_tabtrans, "hyperparameter_selection") 
        path_of_hyperparameter_selection = os.path.join(path_of_tabtrans, "hyperparameter_selection") #path of the folder dataset

        #create the folder for the sample size
        new_folder(path_of_hyperparameter_selection, f"{percentage}")
        path_of_sample = os.path.join(path_of_hyperparameter_selection, f"{percentage}") #path of the folder dataset

        #save a .txt in the folder to save the validation indices
        np.savetxt(os.path.join(path_of_sample, "validation_indices"), val_indices)
        
        experiment_num = 1

        results_table = []
        
        for embd in embed_dim:
            for n_layer in n_layers_lst:
                for n_head in n_heads_lst:
                    try: 
                        print("--------------------------------------------------------------")
                        print(f"NAME: {dataset_name}")
                        print(f"Embedding Size: {embd}")
                        print(f"Number of Layers: {n_layer}")
                        print(f"Number of Heads: {n_head}")

                        #parameters
                        n_layers = n_layer
                        n_heads = n_head
                        embedding_size = embd #The embedding size is set one by one to avoid the out of memory error
                        batch_size = batch_size # 32, 64, 128, 256, 512, 1024
                        epochs = epochs

                        #parameters for the model
                        ff_pw_size = 30  #this value because of the paper 
                        attn_dropout = 0.3 #paper
                        ff_dropout = 0.1 #paper value
                        aggregator = "cls"
                        aggregator_parameters = None
                        decoder_hidden_units = [128,64] #paper value [128,64]
                        decoder_activation_fn = nn.ReLU()
                        need_weights = False
                        numerical_passthrough = False       
                        
                        #experiment i folder
                        new_folder(path_of_sample, f"experiment_{experiment_num}_{embedding_size}_{n_layer}_{n_head}")
                        path_of_experiment = os.path.join(path_of_sample, f"experiment_{experiment_num}_{embedding_size}_{n_layer}_{n_head}") #In this folder it will be saved the model and images

                        #create the folder for the checkpoints
                        new_folder(path_of_experiment, "checkpoints")
                        path_of_checkpoint = os.path.join(path_of_experiment, "checkpoints") #path to save the checkpoints


                        #create the folder for the plots
                        new_folder(path_of_experiment, "plots")
                        path_of_plots = os.path.join(path_of_experiment, "plots") #path of the folder dataset

                        #module
                        module = training.build_module(
                            n_categories, # List of number of categories
                            n_numerical, # Number of numerical features
                            n_heads, # Number of heads per layer
                            ff_pw_size, # Size of the MLP inside each transformer encoder layer
                            n_layers, # Number of transformer encoder layers    
                            n_labels, # Number of output neurons
                            embedding_size,
                            attn_dropout, 
                            ff_dropout, 
                            aggregator, # The aggregator for output vectors before decoder
                            rnn_aggregator_parameters=aggregator_parameters,
                            decoder_hidden_units=decoder_hidden_units,
                            decoder_activation_fn=decoder_activation_fn,
                            need_weights=need_weights,
                            numerical_passthrough=numerical_passthrough
                        ) 

                        #MODEL
                        model = skorch.NeuralNetClassifier(
                            module=module,
                            criterion=torch.nn.CrossEntropyLoss,
                            optimizer=torch.optim.AdamW,
                            device = "cuda" if torch.cuda.is_available() else "cpu",
                            batch_size = batch_size,
                            max_epochs = epochs,
                            train_split=skorch.dataset.ValidSplit(((train_indices, val_indices),)),
                            callbacks=[
                                ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
                                ("duration", skorch.callbacks.EpochTimer()),
                                EpochScoring(scoring='accuracy', name='train_acc', on_train=True), #
                                Checkpoint(dirname = path_of_checkpoint, load_best = True), 
                                EarlyStopping(patience=15)

                            ],
                            optimizer__lr=1e-4,
                            optimizer__weight_decay=1e-4
                        )
                        ''' 
                        model = skorch.NeuralNetClassifier(
                        module=module,
                        criterion=torch.nn.CrossEntropyLoss,
                        optimizer=torch.optim.AdamW,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                        batch_size = batch_size,
                        max_epochs = epochs,
                        train_split=skorch.dataset.ValidSplit(((train_indices, val_indices),)),
                        callbacks=[
                            ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
                            ("accuracy", skorch.callbacks.EpochScoring("accuracy", lower_is_better=False)),
                            ("duration", skorch.callbacks.EpochTimer()),
                            EpochScoring(scoring='accuracy', name='train_acc', on_train=True)
                        ],
                        optimizer__lr=0.5,
                        optimizer__weight_decay=1e-4
                        )
                        
                        
                        model = skorch.NeuralNetClassifier(
                            module=module,
                            criterion=torch.nn.CrossEntropyLoss,
                            optimizer=torch.optim.AdamW,
                            device = "cpu", #if torch.cuda.is_available() else "cpu",
                            batch_size = batch_size,
                            max_epochs = epochs,
                            train_split=skorch.dataset.ValidSplit(((train_indices, val_indices),)),
                            callbacks=[
                                ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
                                ("duration", skorch.callbacks.EpochTimer()),
                                ("accuracy", skorch.callbacks.EpochScoring("accuracy", lower_is_better=False)),
                                EpochScoring(scoring='accuracy', name='train_acc', on_train=True), # Compute accuracy at the end of each epoch on the training set
                                Checkpoint(dirname= path_of_checkpoint,load_best = True),
                                EarlyStopping(patience=20)

                            ],
                            optimizer__lr=1e-4,
                            optimizer__weight_decay=1e-4
                        )
                        '''
                        
                        # Define Checkpoint and TrainEndCheckpoint callbacks with custom directory
                        cp = Checkpoint()
                        train_end_cp = TrainEndCheckpoint()

                        start_time = time.time()  # Start the timer to count how long does it takes
                        #TRAINING
                        model = model.fit(X={
                                "x_numerical": X_train[:, :n_numerical].astype(np.float32),
                                "x_categorical": X_train[:, n_numerical:].astype(np.int32)
                                }, 
                                y=y_train.astype(np.int64)
                                )
                        end_time = time.time()  # Stop the timer
                        training_time = end_time - start_time  # Calculate the elapsed time
                        
                        #TESTING
                        #I will use only the valdation indices to have the model selection 


                        predictions = model.predict_proba(X={
                                        "x_numerical": X_train[:, :n_numerical].astype(np.float32),
                                        "x_categorical": X_train[:, n_numerical:].astype(np.int32)
                                        }
                                        )
                                
                        
                        print("Test results in validation:\n")
                        metrics = evaluating.get_default_scores(y_train[val_indices].astype(np.int64), predictions[val_indices], multiclass = multiclass_val)
                        print(metrics)

                        # Assign dictionary keys as variable names
                        for key, value in metrics.items(): # Loop through the dictionary
                            globals()[key] = value
                        
                        #create the table to save results
                        columns_names = ["experiment_num", "n_layers", "n_heads", "embed_dim", "batch_size", "max_epochs", "time_trainning"]
                        columns_names.extend(metrics.keys()) #add the keys of the metrics

                        print(columns_names)

                        max_epochs = len(model.history)
                        result_row = [experiment_num, n_layers, n_heads, embd, batch_size, max_epochs, training_time]
                        result_row.extend(metrics.values()) #add the values of the metrics

                        results_table.append(result_row)




                        #print(evaluating.get_default_scores(y_train[val_indices].astype(np.int64), predictions[val_indices], multiclass = multiclass_val))
                        
                        #balanced_accuracy = evaluating.get_default_scores(y_train[val_indices].astype(np.int64), predictions[val_indices], multiclass = multiclass_val)["balanced_accuracy"]
                        #accuracy = evaluating.get_default_scores(y_train[val_indices].astype(np.int64), predictions[val_indices], multiclass = multiclass_val)["accuracy"]
                        #log_loss = evaluating.get_default_scores(y_train[val_indices].astype(np.int64), predictions[val_indices], multiclass = multiclass_val)["log_loss"]

                        #save the results in a list
                        #result_row = [dataset_name, experiment_num, n_layers, n_heads, embd, batch_size, balanced_accuracy, accuracy, log_loss, max_epochs, training_time]
                        #results_table.append(result_row)

                        #create and save the plots
                        fig = plots.model_plots(model, f"Experiment {experiment_num}")

                        # Save the first figure
                        fig.savefig(os.path.join(path_of_plots, 'plots.png'))
                        #increase experiment number, this one is used for the plots
                        
                        experiment_num += 1

                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"CUDA out of memory")
                            torch.cuda.empty_cache()  # Free up memory if possible               


        export_to_csv(results_table, columns_names, path_of_sample)


def train_xgboost(task_id, sample_size, project_path):
    for percentage in sample_size:
        
        #From Id get the data
        #X_train, X_test, y_train, y_test, train_indices, val_indices, _, n_labels, n_numerical, n_categories = data.import_data(task_id, percentage)

        #columns_names = ["dataset_name", "experiment_num", "balanced_accuracy", "accuracy", "log_loss", "max_epochs", "time_trainning"]
        #results_table = []

        #save a .txt in the folder to save the validation indices
        #np.savetxt(os.path.join(path_of_sample, "validation_indices"), val_indices)

        #get the dataset_name
        if task_id in [1484,1564]: #two selected datasets with no task id, just id
            dataset = openml.datasets.get_dataset(task_id)
            dataset_name = dataset.name
        else:
            dataset_name = data.get_dataset_name(task_id)

        #create the folder to save the dataset experiments if it doesn't exist
        new_folder(project_path, "Final_models_2")
        path_of_data_models = os.path.join(project_path, "Final_models_2") #path of the folder data_models

        #create the folder for specific dataset name
        new_folder(path_of_data_models, f"{dataset_name}") #_{embed_dim}") #
        path_of_dataset = os.path.join(path_of_data_models, f"{dataset_name}") #path of the folder dataset 

        #create the xgboost folder
        new_folder(path_of_dataset, "xgboost") 
        path_of_xgboost = os.path.join(path_of_dataset, "xgboost") #path of the folder dataset

        #create folder of hyperparameter selection
        new_folder(path_of_xgboost, "hyperparameter_selection") 
        path_of_hyperparameter_selection = os.path.join(path_of_xgboost, "hyperparameter_selection") #path of the folder dataset

        #create the folder for the sample size
        new_folder(path_of_hyperparameter_selection, f"{percentage}")
        path_of_sample = os.path.join(path_of_hyperparameter_selection, f"{percentage}") #path of the folder dataset
        
        best_params, metrics, cv_results = xgboost_file.run_xgboost(task_id, percentage)

        #Let's export this 3 dicts to csv

        # Open the file in write mode and export the dictionary as CSV
        path_parm = os.path.join(path_of_sample, "parameters.csv")

        with open(path_parm, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header (keys)
            writer.writerow(best_params.keys())
            
            # Write values
            writer.writerow(best_params.values())  
        
        
        path_metrics = os.path.join(path_of_sample, "metrics.csv")

        with open(path_metrics, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header (keys)
            writer.writerow(metrics.keys())
            
            # Write values
            writer.writerow(metrics.values())


        path_cv = os.path.join(path_of_sample, "cross_validation.csv")

        with open(path_cv, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # Write header (keys)
            writer.writerow(cv_results.keys())
            
            # Write values
            writer.writerow(cv_results.values())
        








