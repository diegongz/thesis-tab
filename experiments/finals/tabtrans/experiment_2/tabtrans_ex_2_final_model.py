'''
In this file we will import the best configuration hyperparameters and train the model with them.
'''

import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of this file

# Get the project directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_folder)))) #parent folder of current_folder

#print(project_path)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files

from utils import data, tabtrans_file, plots
import pandas as pd
import numpy as np


df_id = 1484

name_df = data.get_dataset_name(df_id)

path_of_datset = f'{project_path}/Final_models_3/{name_df}' #The path can be 

path_to_hyperparameters = f'{path_of_datset}/tabtrans/hyperparameter_selection'

#define the path to final_tabtrans
path_to_final_tabtrans = f'{path_of_datset}/tabtrans/final_tabtrans'

#create the directory if it does not exist
os.makedirs(path_to_final_tabtrans, exist_ok=True)

sample_sizes = [80] # 100, 80, 60, 40, 20

for sample in sample_sizes:
    path_of_size = f'{path_to_hyperparameters}/{sample}'

    if sample == 100:
        
        #Import the data
        X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

        path_of_results = f'{path_of_size}/results.csv'

        hyperparameters = data.import_hyperparameters(path_of_results)

        n_layers = int(hyperparameters["n_layers"])
        n_heads = int(hyperparameters["n_heads"])
        embedding_size = int(hyperparameters["embedding_size"])
        batch_size = int(hyperparameters["batch_size"])
        epochs = int(hyperparameters["max_epochs"])

        model, metrics = tabtrans_file.general_tabtrans(X_train, X_test, y_train, y_test, train_indices, val_indices, n_labels, n_numerical, n_categories, n_layers, n_heads, embedding_size, batch_size, epochs, hyperparameter_search = False)


        metrics["n_layers"] = n_layers
        metrics["n_heads"] = n_heads
        metrics["embedding_size"] = embedding_size
        metrics["batch_size"] = batch_size
        metrics["max_epochs"] = epochs

        # Convert dictionary to DataFrame
        df = pd.DataFrame(metrics, index=[0])

        #folder of the size
        final_tab_size_path = f'{path_to_final_tabtrans}/{sample}' 

        #create the directory if it does not exist
        os.makedirs(final_tab_size_path, exist_ok=True)

        final_result_path = f'{final_tab_size_path}/results.csv'

        # Convert the DataFrame to a CSV file
        df.to_csv(final_result_path, index=False)
    
        print(f"All Seeds have been trained for sample size:{sample}")


    else:
        path_of_trials = f'{path_to_hyperparameters}/{sample}/trials'

        seed = 11  # You can choose any integer value as the seed
        np.random.seed(seed)

        # Generate 10 random integers between a specified range (e.g., 0 to 100)
        random_seeds = np.random.randint(0, 100, size=10)

        number_of_seed = 1

        path_of_results = f'{path_of_trials}/seed_1/results.csv'

        hyperparameters = data.import_hyperparameters(path_of_results)

        n_layers = int(hyperparameters["n_layers"])
        n_heads = int(hyperparameters["n_heads"])
        embedding_size = int(hyperparameters["embedding_size"])
        batch_size = int(hyperparameters["batch_size"])
        epochs = int(hyperparameters["max_epochs"])

        for random_seed in random_seeds:

            X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

            train_indices, val_indices = data.reduce_size(y_train, train_indices, val_indices, sample, random_seed)

            model, metrics = tabtrans_file.general_tabtrans(X_train, X_test, y_train, y_test, train_indices, val_indices, n_labels, n_numerical, n_categories, n_layers, n_heads, embedding_size, batch_size, epochs, hyperparameter_search = False)

            metrics["n_layers"] = n_layers
            metrics["n_heads"] = n_heads
            metrics["embedding_size"] = embedding_size
            metrics["batch_size"] = batch_size
            metrics["max_epochs"] = epochs

            # Convert dictionary to DataFrame
            df = pd.DataFrame(metrics, index=[0])

            final_tab_size_path = f'{path_to_final_tabtrans}/{sample}/trials'
            os.makedirs(final_tab_size_path, exist_ok=True)

            final_model_seed_path = f'{final_tab_size_path}/seed_{number_of_seed}'
            os.makedirs(final_model_seed_path, exist_ok=True)

            final_result_path = f'{final_model_seed_path}/results.csv'

            # Convert the DataFrame to a CSV file
            df.to_csv(final_result_path, index=False)

            number_of_seed += 1

        print(f"All Seeds have been trained for sample size:{sample}")
