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
from utils import training, callback, evaluating, attention, data, plots, fast_model
from sklearn import datasets, model_selection
import skorch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import openml
from sklearn import datasets, model_selection
from skorch.callbacks import Checkpoint, EarlyStopping, LoadInitState, EpochScoring, Checkpoint, TrainEndCheckpoint
import csv
import time


def final_tab_trans(ds_id, sample_size, project_path):

    #Import data
    X_train, X_test, y_train, y_test, train_indices, val_indices, _, n_labels, n_numerical, n_categories = data.import_data(ds_id, sample_size)

    model_name = "tabtrans"

    #name of the dataset
    if ds_id in [1484,1564]: #two selected datasets with no task id, just id
        dataset = openml.datasets.get_dataset(ds_id)
        dataset_name = dataset.name
    
    else:
        dataset_name = data.get_dataset_name(ds_id)

    #Find if I have multiclass in the y's
    if len(np.unique(y_train)) > 2:
        multiclass_val = True
    
    else:
        multiclass_val = False

    #Load hyperparameters
    hyperparameters = data.import_hyperparameters(ds_id, sample_size, model_name, project_path)

    #Extract the hyperparameters
    n_heads = hyperparameters["n_heads"]
    embed_dim = hyperparameters["embed_dim"]
    n_layers = hyperparameters["n_layers"]
    ff_pw_size = 30  #this value because of the paper
    attn_dropout = 0.3 #paper
    ff_dropout = 0.1 #paper value
    aggregator = "cls"
    aggregator_parameters = None
    decoder_hidden_units = [128,64] #paper value
    decoder_activation_fn = nn.ReLU()
    need_weights = False
    numerical_passthrough = False

    batch_size = hyperparameters["batch_size"]
    epochs = hyperparameters["max_epochs"]

    """
    Building PyTorch module.

    We provide a wrapper function for building the PyTorch module.
    The function is utils.training.build_module.
    """
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

    #create the model
    model = skorch.NeuralNetClassifier(
            module = module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            device= "cuda", #cuda" if torch.cuda.is_available() else
            batch_size = batch_size,
            train_split = None,
            max_epochs = epochs,
            optimizer__lr=1e-4,
            optimizer__weight_decay=1e-4
        )

    start_time = time.time()  # Start the timer to count how long does it takes
    try: 
        #model creation
        model = model.fit(X={
            "x_numerical": X_train[:, :n_numerical].astype(np.float32),
            "x_categorical": X_train[:, n_numerical:].astype(np.int32)
            }, 
            y=y_train.astype(np.int64)
        )
    
    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print(f"CUDA out of memory")
                            torch.cuda.empty_cache()  # Free up memory if possible
                        else:
                            print("There was an error")
    

    end_time = time.time()  # Stop the timer
    training_time = end_time - start_time  # Calculate the elapsed time


    #predictions
    predictions = model.predict_proba(X={
    "x_numerical": X_test[:, :n_numerical].astype(np.float32),
    "x_categorical": X_test[:, n_numerical:].astype(np.int32)
    }
    )

    print("Test results:\n")

    metrics = evaluating.get_default_scores(y_test.astype(np.int64), predictions, multiclass = multiclass_val)
    print(metrics)

    #define all the keys equals value
    
    # Assign dictionary keys as variable names
    for key, value in metrics.items(): # Loop through the dictionary
        globals()[key] = value

    #create the table to save results
    columns_names = ["n_layers", "n_heads", "embed_dim", "batch_size", "max_epochs", "time_trainning"]
    columns_names.extend(metrics.keys()) #add the keys of the metrics

    results_table = []

    #results row
    result_row = [n_layers, n_heads, embed_dim, batch_size, epochs, training_time]
    result_row.extend(metrics.values()) #add the values of the metrics

    results_table.append(result_row)



    model_name_path = os.path.join(project_path, "Final_models", dataset_name, model_name)
    
    #create the folder final_tabtrans
    fast_model.new_folder(model_name_path, "final_tabtrans")

    #path to the final_tabtrans folder
    final_tabtrans_folder = os.path.join(model_name_path, "final_tabtrans")

    #create the folder for the sample size
    fast_model.new_folder(final_tabtrans_folder, f"{sample_size}")

    #path to sample size folder
    sample_size_folder = os.path.join(final_tabtrans_folder, f"{sample_size}") #here i will save the results

    #create the table to save results
    #columns_names = ["n_layers", "n_heads", "embed_dim", "batch_size", "balanced_accuracy", "accuracy", "log_loss", "max_epochs", "time_trainning"]
    #results_table = []

    #balanced_accuracy = evaluating.get_default_scores(y_test.astype(np.int64), predictions, multiclass = multiclass_val)["balanced_accuracy"]
    #accuracy = evaluating.get_default_scores(y_test.astype(np.int64), predictions, multiclass = multiclass_val)["accuracy"]
    #log_loss = evaluating.get_default_scores(y_test.astype(np.int64), predictions, multiclass = multiclass_val)["log_loss"]


    fast_model.export_to_csv(results_table, columns_names, sample_size_folder)









