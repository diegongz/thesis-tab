''' 
In this file we will compute all the epochs and save those models, then i will compute
the entropy for all epochs and plot the train and validation entropy with the loss and accuracy
''' 
import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of this file

# Get the project directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_folder)))) #parent folder of current_folder

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files


from utils import data, tabtrans_file, plots,attention, training, attention_file
import numpy as np
import matplotlib.pyplot as plt
from skorch.callbacks import TrainEndCheckpoint, EarlyStopping, LoadInitState, Checkpoint, EpochScoring
import skorch
import torch.nn as nn
import torch
from scipy.stats import entropy
import csv



'''
FINAL DATASETS
1484 lsvt
31 credit 
12 mfeat-factors 217 pass
9964 semeion 257 pass
233092 arrhythmia 280
3485 scene 300
41966 isolet (600x618)
2 anneal (898x39)
'''


datasets = [2] #1484, 31, 12, 20, 9964, 233092, 3485, 41966, 2


pairs_loss_entropy = [] #here i will save the pairs of loss and entropy

for df_id in datasets:
    sample = 100

    X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)


    n_features = X_train.shape[1]
    name_df = data.get_dataset_name(df_id)

    path_of_datset = f'{project_path}/Final_models_4/{name_df}'
    path_to_hyperparameters = f'{path_of_datset}/tabtrans/hyperparameter_selection'

    path_of_hyper_size = f'{path_to_hyperparameters}/{sample}'
    path_of_hyper_results = f'{path_of_hyper_size}/results.csv'


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

    #import the hyperparameters
    hyperparameters = data.import_hyperparameters(path_of_hyper_results, cv = True)

    n_layers = int(hyperparameters["n_layers"])
    n_heads = int(hyperparameters["n_heads"])
    embedding_size = int(hyperparameters["embedding_size"])
    batch_size = int(hyperparameters["batch_size"])
    epochs = int(hyperparameters["max_epochs_mean"])

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


    path_to_checkpoint = f"{path_of_datset}/entropy/checkpoints" #create the path to save the checkpoint
    os.makedirs(path_to_checkpoint, exist_ok = True)

    epoch_num = 1

    train_end_cp = TrainEndCheckpoint(dirname = f"{path_to_checkpoint}/epoch_{epoch_num}")


    model = skorch.NeuralNetClassifier(
        module = module,
        criterion=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.AdamW,
        device= "cuda", #cuda" if torch.cuda.is_available() else
        batch_size = batch_size,
        max_epochs = epoch_num,
        train_split=skorch.dataset.ValidSplit(((train_indices, val_indices),)),
        optimizer__lr=1e-4,
        optimizer__weight_decay=1e-4,
        callbacks=[train_end_cp,
                ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
                EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
                ]
        )

    model = model.fit(X={
        "x_numerical": X_train[:, :n_numerical].astype(np.float32),
        "x_categorical": X_train[:, n_numerical:].astype(np.int32)
        }, 
        y=y_train.astype(np.int64)     
        )

    total_epochs = 100
    counter = 1
        
    while counter < total_epochs:
            
        load_state = LoadInitState(train_end_cp) #load the state of the past model
        train_end_cp = TrainEndCheckpoint(dirname = f"{path_to_checkpoint}/epoch_{counter+1}")
        
        #create the model
        model = skorch.NeuralNetClassifier(
            module = module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            device= "cuda", #cuda" if torch.cuda.is_available() else
            batch_size = batch_size,
            max_epochs = 1,
            train_split =skorch.dataset.ValidSplit(((train_indices, val_indices),)),
            optimizer__lr=1e-4,
            optimizer__weight_decay=1e-4,
            callbacks=[train_end_cp, load_state,
                    ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
                    EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
                    ]
            )

        model = model.fit(X={
            "x_numerical": X_train[:, :n_numerical].astype(np.float32),
            "x_categorical": X_train[:, n_numerical:].astype(np.int32)
            }, 
            y=y_train.astype(np.int64)
            )
        
        counter += 1

    #Save the epochs, val_loss, train_loss, val_acc, train_acc
    epochs = []

    train_acc = []
    val_acc = []

    train_loss = []
    val_loss = []

    for x in model.history:
        epoch_num = x["epoch"]
        epochs.append(epoch_num)

        train_acc.append(x['train_acc'])
        val_acc.append(x['valid_acc'])

        train_loss.append(x["train_loss"])
        val_loss.append(x["valid_loss"])
        

    epochs = list(range(1, total_epochs+1))
    epoch_average_entropy = []

    sample_size = 100

    epoch_avg_entropy = []

    for epoch in epochs:    
        #DEFINE THE MODEL EXACTLY AS BEFORE
        
        model = skorch.NeuralNetClassifier(
            module = module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            device= "cuda", #cuda" if torch.cuda.is_available() else
            batch_size = batch_size,
            max_epochs = 1,
            train_split =skorch.dataset.ValidSplit(((train_indices, val_indices),)),
            optimizer__lr=1e-4,
            optimizer__weight_decay=1e-4,
            callbacks=[train_end_cp, load_state,
                    ("balanced_accuracy", skorch.callbacks.EpochScoring("balanced_accuracy", lower_is_better=False)),
                    EpochScoring(scoring='accuracy', name='train_acc', on_train=True),
                    ]
            )
        
        #Let's define the path where the model parameters are stored for an specific epoch
        path_to_epoch_model = f'{path_to_checkpoint}/epoch_{epoch}/train_end_params.pt'
        
        model.initialize()
        model.load_params(f_params = path_to_epoch_model) #Here the model is loaded
        
        #compute the attention matrix
        matrix = attention_file.attention_matrix(model, X_train[train_indices], y_train[train_indices], n_numerical, n_layers, n_heads, n_features+1)

        entropy_per_row = np.apply_along_axis(entropy, 1, matrix, base=2) / np.log2(n_features+1)
        average_entropy = np.mean(entropy_per_row)

        epoch_avg_entropy.append(average_entropy)



    for i in range(len(epochs)):
        pair = []
        pair.append(val_loss[i])
        pair.append(epoch_avg_entropy[i])
        
        pairs_loss_entropy.append(pair)
        
        
    
          
    index = data.find_no_decrease_index(epoch_avg_entropy)

    if index != -1:
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("\n")
        print("\n")
        print("\n")
        print(f"The entropy started to decrease at epoch {index}")
        print("\n")
        print("\n")
        print("\n")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    
    else:
        print("----------------------------------------------------------------------------")
        print("\n")
        print("\n")
        print("\n")
        print(f"Non early stopping needed for {name_df} using entropy")
        print("\n")
        print("\n")
        print("\n")
        print("----------------------------------------------------------------------------")

    #Save the plots
    plots_path = f"{path_of_datset}/entropy/plots" #create the path to save the checkpoint

    os.makedirs(plots_path, exist_ok = True)


    path_loss = f"{plots_path}/loss.png"
    plots.three_lines_plot(epochs, train_loss, val_loss, epoch_avg_entropy, "train", "validation", "entropy", f"Loss {name_df}", path_loss)

    path_acc = f"{plots_path}/accuracy.png"
    plots.three_lines_plot(epochs, train_acc, val_acc, epoch_avg_entropy, "train", "validation", "entropy", f"Balanced Accuracy {name_df}", path_acc)

    path_entropy = f"{plots_path}/entropy.png"
    #plot average entropy
    # Example data

    # Create the plot with a green line
    plt.plot(epochs, epoch_avg_entropy, color='green')

    # Add labels and title (optional)
    plt.xlabel("Epochs")
    plt.ylabel("Entropy")
    plt.title(f"Entropy {name_df}")
    plt.savefig(path_entropy, dpi=300)

    

# Specify the file path
output_path = f"{project_path}/Final_models_4/all_datasets/sample_entropy.csv"

# Write the list of pairs to a CSV file
with open(output_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["val_loss", "entropy"])  # Write header
    writer.writerows(pairs_loss_entropy)       # Write data rows

print(f"Data successfully saved to {output_path}")