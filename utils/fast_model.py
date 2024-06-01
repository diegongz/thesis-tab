import sys
project_path = "/home/diego/Git/thesis-tabtrans"
sys.path.append(project_path) #import folders from the project_path

import os
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
import csv

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

def model_creation(parameters, project_path):
    
    #extract parameters
    task_id = parameters["task_id"]
    layers = parameters["Layers"]
    heads = parameters["Heads"]
    embed_dim = parameters["Emedding_dim"]
    batch_size = parameters["batch_size"]
    epochs = parameters["epochs"]
    
    columns_names = ["dataset_name", "experiment_num", "n_layers", "n_heads", "embed_dim", "batch_size", "balanced_accuracy", "accuracy", "log_loss", "epochs", "time"]
    results_table = []


    #get the dataset_name
    dataset_name = data.get_dataset_name(task_id)

    X_train, X_test, y_train, y_test, train_indices, val_indices, _, n_labels, n_numerical, n_categories = data.import_data(task_id)

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

    if len(np.unique(y_train)) > 2:
        multiclass_val = True
    
    else:
        multiclass_val = False
    

    #create the folder to save the dataset experiments if it doesn't exist
    new_folder(project_path, "data_models")
    path_of_data_models = os.path.join(project_path, "data_models") #path of the folder data_models

    #create the folder for specific dataset name
    new_folder(path_of_data_models, f"{dataset_name}_{embed_dim}")
    path_of_dataset = os.path.join(path_of_data_models, f"{dataset_name}_{embed_dim}") #path of the folder dataset 

    #save a .txt in the folder to save the validation indices
    np.savetxt(os.path.join(path_of_dataset, "validation_indices"), val_indices)
    
    experiment_num = 1
    
    for n_layers in layers:
        for n_heads in heads:
            #experiment i folder
            new_folder(path_of_dataset, f"experiment_{experiment_num}")
            path_of_experiment = os.path.join(path_of_dataset, f"experiment_{experiment_num}") #In this folder it will be saved the model and images


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
                embed_dim,
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
            '''
             
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
            
            # Define Checkpoint and TrainEndCheckpoint callbacks with custom directory
            #cp = Checkpoint()
            #train_end_cp = TrainEndCheckpoint()


            #TRAINING
            model = model.fit(X={
                    "x_numerical": X_train[:, :n_numerical].astype(np.float32),
                    "x_categorical": X_train[:, n_numerical:].astype(np.int32)
                    }, 
                    y=y_train.astype(np.int64)
                    )
            
            #TESTING
            predictions = model.predict_proba(X={
                            "x_numerical": X_test[:, :n_numerical].astype(np.float32),
                            "x_categorical": X_test[:, n_numerical:].astype(np.int32)
                            }
                            )
            
            print("Test results:\n")
            print(evaluating.get_default_scores(y_test.astype(np.int64), predictions, multiclass = multiclass_val))
            
            balanced_accuracy = evaluating.get_default_scores(y_test, predictions, multiclass = multiclass_val)["balanced_accuracy"]
            accuracy = evaluating.get_default_scores(y_test, predictions, multiclass = multiclass_val)["accuracy"]
            log_loss = evaluating.get_default_scores(y_test, predictions, multiclass = multiclass_val)["log_loss"]

            #save the results in a list
            result_row = [dataset_name, experiment_num, n_layers, n_heads, embed_dim, batch_size, balanced_accuracy, accuracy, log_loss]
            results_table.append(result_row)

            #create and save the plots
            fig_1, fig_2 = plots.model_plots(model, f"Experiment {experiment_num}")

            # Save the first figure
            fig_1.savefig(os.path.join(path_of_plots, 'figure1.png'))

            # Save the second figure
            fig_2.savefig(os.path.join(path_of_plots, 'figure2.png'))

            #increase experiment number, this one is used for the plots
            experiment_num += 1


    export_to_csv(results_table, columns_names, path_of_dataset)