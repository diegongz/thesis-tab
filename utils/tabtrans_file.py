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

'''
X_train: Training data from 90-10 X
X_test: 10% from the first split X
y_train: Training data from 90-10 y
y_test: 10% from the first split y
train_indices: The indices that will be used for training X_train[train_indices] is the original train
val_indices: The indices that will be used for validaiton X_train[val_indices] is the validation sets
n_labels: Number of classes
n_numerical: Number of numerical features
n_categories: List that tells mes how many categories every categorical feature has 
dataset_name: The name of the dataset
n_layers: Number of layers that the transformer will set
n_heads: number of heads
embedding_size: embedding size used for the embeddings
batch_size: size of the batches
epochs: max number of epochs
hyperparameter_search: By default is False, which mean will train a normal model (without doing validation), if True then will use the validation set to train the best model
'''
def general_tabtrans(X_train, X_test, y_train, y_test, train_indices, val_indices, n_labels, n_numerical, n_categories, n_layers, n_heads, embedding_size, batch_size, epochs, hyperparameter_search = False):

    #Find if I have multiclass in the y's
    if len(np.unique(y_train)) > 2:
        multiclass_val = True
    
    else:
        multiclass_val = False

    print(f"Embedding Size: {embedding_size}")
    print(f"Number of Layers: {n_layers}")
    print(f"Number of Heads: {n_heads}")

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

    if hyperparameter_search == True:
        #MODEL
        model = skorch.NeuralNetClassifier(
            module=module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            device = "cuda", #if torch.cuda.is_available() else "cpu",
            batch_size = batch_size,
            max_epochs = epochs,
            train_split=skorch.dataset.ValidSplit(((train_indices, val_indices),)),
            callbacks=[
                ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
                ("duration", skorch.callbacks.EpochTimer()),
                EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
                #Checkpoint(dirname = path_of_checkpoint, load_best = True), 
                EarlyStopping(patience=15)

            ],
            optimizer__lr=1e-4,
            optimizer__weight_decay=1e-4
        )
        
    else:
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
    
    #TRAINING
    model = model.fit(X={
            "x_numerical": X_train[:, :n_numerical].astype(np.float32),
            "x_categorical": X_train[:, n_numerical:].astype(np.int32)
            }, 
            y=y_train.astype(np.int64)
            )
    end_time = time.time()  # Stop the timer
    training_time = end_time - start_time  # Calculate the elapsed time    
    
    if hyperparameter_search == True:       
        predictions = model.predict_proba(X={
                                            "x_numerical": X_train[:, :n_numerical].astype(np.float32),
                                            "x_categorical": X_train[:, n_numerical:].astype(np.int32)
                                            }
                                            )
        metrics = evaluating.get_default_scores(y_train[val_indices].astype(np.int64), predictions[val_indices], multiclass = multiclass_val)
    
    else:
        #predictions
        predictions = model.predict_proba(X={
        "x_numerical": X_test[:, :n_numerical].astype(np.float32),
        "x_categorical": X_test[:, n_numerical:].astype(np.int32)
        }
        )

        metrics = evaluating.get_default_scores(y_test.astype(np.int64), predictions, multiclass = multiclass_val)

        #add the training_time as a value in metrics dictionary
        metrics["time_trainning"] = training_time

    return model, metrics



#-----------------------------FINAL TABTRANS--------------------------------------------
'''
This function returns the final tabtrans model metrics using the best hyperparameters configuration

hyperparameters: Dictionary with the hyperparameters configuration
'''

def final_tab_trans(ds_id, X_train, X_test, y_train, y_test, train_indices, val_indices, n_labels, n_numerical, n_categories, hyperparameters):

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
    hyperparameters = data.import_hyperparameters(ds_id, sample_size, model_name, project_path, name_folder_models)

    #Extract the hyperparameters
    n_heads = int(hyperparameters["n_heads"])
    embed_dim = int(hyperparameters["embed_dim"])
    n_layers = int(hyperparameters["n_layers"])
    ff_pw_size = 30  #this value because of the paper
    attn_dropout = 0.3 #paper
    ff_dropout = 0.1 #paper value
    aggregator = "cls"
    aggregator_parameters = None
    decoder_hidden_units = [128,64] #paper value
    decoder_activation_fn = nn.ReLU()
    need_weights = False
    numerical_passthrough = False

    batch_size = int(hyperparameters["batch_size"])
    epochs = int(hyperparameters["max_epochs"])
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
    
    #model creation
    model = model.fit(X={
        "x_numerical": X_train[:, :n_numerical].astype(np.float32),
        "x_categorical": X_train[:, n_numerical:].astype(np.int32)
        }, 
        y=y_train.astype(np.int64)
    )
    '''
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
    

    '''
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



    model_name_path = os.path.join(project_path, f"{name_folder_models}", dataset_name, model_name)
    
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









