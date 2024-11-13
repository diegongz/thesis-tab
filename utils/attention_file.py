import numpy as np
from utils import attention, data, training
import skorch
import torch.nn as nn
import torch
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from scipy.stats import entropy


"""
Eneabling and extracting the attention cubes.

To eneable the attention cubes recovering, the only requirement is to 
set the PyTorch module need_weights=True. When the cubes are required the
new output will be:

    - predictions: The predictionsfor the given instances
    - layer outputs: The output of each encoder layer
    - weights: The attention cube of each encoder

In skorch, the trained PyTorch module is saved in the variable .module_.

When using skorch, the only way to recover multiple outputs is by
using the forward/forward_iter method.
"""

'''
model: should be of type <class 'skorch.classifier.NeuralNetClassifier'>
'''
def attention_matrix(model, X_train, y_train, n_numerical, n_layers, n_heads, n_features):
    model.module_.need_weights = True
    cumulative_attns = []

    for X_inst, y_inst in zip(X_train, y_train):
        pred, layer_outputs, attn = model.forward(X={
            "x_numerical": X_inst[None, :n_numerical].astype(np.float32),
            "x_categorical": X_inst[None, n_numerical:].astype(np.int32)
            })
            
        """
        The attention cubes dimensions are:
        
        (num. layers, batch size, num. heads, num. features, num. features)
        #Why does the batch size is 1?
        """
        assert attn.shape == (n_layers, 1, n_heads, n_features, n_features) 
        
        """
        To compute the cumulative attention we provide a function in:
        
            utils.attention.compute_std_attentions(attention, aggregator)
            
        The function returns:
            The inidivual attention (non cumulative) of each layer. Shape:  (num layers, batch size, num. features)
            The cumulative attention at each layer. Shape: (num layers, batch size, num. features)
            
        The last layerof the cumulative attention represents the cumulative attention over all
        Transformer Encoders.
        """
        aggregator = "cls" #for this case we use the aggregation of the [CLS] token but it can be changed
        
        ind_attn, cum_attn = attention.compute_std_attentions(attn, aggregator)
        
        assert ind_attn.shape == (n_layers, 1, n_features)
        assert cum_attn.shape == (n_layers, 1, n_features)
        
        cumulative_attns.append(cum_attn[-1, 0])
    
    cumulative_attns = np.array(cumulative_attns)
    
    return cumulative_attns

def average_entropy_matrix(matrix):
    """
    This function receives a matrix of shape (n_instances, n_features) and returns the average entropy of the matrix.
    """
    n_features = matrix.shape[1]
    
    # Compute the entropy for rows
    entropy_per_row = np.apply_along_axis(entropy, 1, attention_matrix, base=2) / np.log2(n_features+1) 
    
    # Compute the average entropy
    average_entropy = np.mean(entropy_per_row)
    
    return average_entropy

''' 
Function that computes the attention matrix and then returns the average entropy of that matrix
'''
def entropy_attention_matrix(net, X_train, y_train, n_numerical, n_layers, n_heads, n_features):
    matrix = attention_matrix(net, X_train, y_train, n_numerical, n_layers, n_heads, n_features)
    
    print("Model Output Shape:", net.predict(X_train).shape)
    print("Target Labels Shape:", y_train.shape)
    
    return average_entropy_matrix(matrix)

'''
Given a df_id and an specific epoch this function will return the attention matrix
It will load the model and the 
'''
def matrix_for_epoch(df_id, epoch, sample_size, project_path):
    
    name_df = data.get_dataset_name(df_id)
    
    path_of_datset = f'{project_path}/Final_models_4/{name_df}'
    
    #We need to extract the hyperparameters that were used to train the model
    path_to_hyperparameters = f'{path_of_datset}/tabtrans/hyperparameter_selection'
    
    #define the path to final_tabtrans
    path_to_final_tabtrans = f'{path_of_datset}/tabtrans/final_tabtrans_cv'
    
    #import the data
    X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

    #Let's import the data and the hyperparameters
    path_hyperparameter_sample = f'{path_to_hyperparameters}/{sample_size}'
    path_of_hyperparameter_file = f'{path_hyperparameter_sample}/results.csv' #Here The hyperparameters are stored
    
    
    #Import the hyperparameters that were used to train the final model
    hyperparameters = data.import_hyperparameters(path_of_hyperparameter_file, cv = True)
    n_layers = int(hyperparameters["n_layers"])
    n_heads = int(hyperparameters["n_heads"])
    embedding_size = int(hyperparameters["embedding_size"])
    batch_size = int(hyperparameters["batch_size"])
    epochs = int(hyperparameters["max_epochs_mean"])
    n_features = X_train.shape[1]+1
    
    #define the module of the model that will be used to load the model
    
    #parameters of the NN
    ff_pw_size = 30  #this value because of the paper
    attn_dropout = 0.3 #paper
    ff_dropout = 0.1 #paper value
    aggregator = "cls"
    aggregator_parameters = None
    decoder_hidden_units = [128,64] #paper value
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

    #Define the model exactly the same way as it was trained
    model = skorch.NeuralNetClassifier(
            module = module,
            criterion=torch.nn.CrossEntropyLoss,
            optimizer=torch.optim.AdamW,
            device= "cuda", #cuda" if torch.cuda.is_available() else
            batch_size = batch_size,
            train_split = None,
            optimizer__lr=1e-4,
            optimizer__weight_decay=1e-4,
        )
    
    #Let's define the path where the model parameters are stored for an specific epoch
    path_to_epoch_model = f'{path_of_datset}/tabtrans/final_tabtrans_cv/{sample_size}/checkpoints/epoch_{epoch}/train_end_params.pt'

    # Initialize the model
    model.initialize()
    model.load_params(f_params = path_to_epoch_model) #Here the model is loaded
    
    matrix = attention_matrix(model, X_train, y_train, n_numerical, n_layers, n_heads, n_features)
    
    return matrix

def heatmap_matrix(cumulative_attns, df_name, save_path):
    """
    Each row represents a test instance, while each column
    represents a feature.
    """
    kmeans = KMeans(n_clusters=4, random_state=11)
    kmeans.fit(cumulative_attns)
    
    #get the labels
    cluster_labels = kmeans.labels_
    
    #sortd indices based on cluster labers
    sorted_indices = np.argsort(cluster_labels) #returns the indices that would sort the array cluster_labels
    
    #sorted matrix by labels
    sorted_cumulative_attns = cumulative_attns[sorted_indices]
    
    #Now order the cluster labels based on the sorted indices to have the instance ordered, and the clusters ordered and also the labels
    sorted_cluster_labels = cluster_labels[sorted_indices]
    
    #Here I will save the index of the last element of each cluster
    last_cluster_indices = []

    for i in range(len(sorted_cluster_labels)-1):
        actual_value = sorted_cluster_labels[i]
        next_value = sorted_cluster_labels[i+1]
        
        if actual_value != next_value:
            last_cluster_indices.append(i)
    
    
    len_cumm_vector = sorted_cumulative_attns.shape[1]

    # last_cluster_indices is a list of indices where you want to add NaN rows
    nan_row = np.full(len_cumm_vector, np.nan)

    # Insert NaN rows
    #It does it in reverse in order to not change the indices of the elements
    for i in sorted(last_cluster_indices, reverse=True):
        sorted_cumulative_attns = np.insert(sorted_cumulative_attns, i + 1, nan_row, axis=0) #insert the NaN array row in the sorted_cumulative_attns
        
        
    fig = px.imshow(sorted_cumulative_attns, color_continuous_scale='Inferno', aspect='auto', origin='lower', title = f'Attention Matrix for {df_name}')

    # Save the figure as a PNG
    fig.write_image(save_path, scale=3)
    
    
def variation_distance(P, Q):
    return 0.5 * np.sum(np.abs(P - Q))