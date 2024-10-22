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

df_id = 31

name_df = data.get_dataset_name(df_id)

path_of_datset = f'{project_path}/Final_models_4/{name_df}' #The path can be 

path_to_hyperparameters = f'{path_of_datset}/tabtrans/hyperparameter_selection'

#define the path to final_tabtrans
path_to_final_tabtrans = f'{path_of_datset}/tabtrans/final_tabtrans_cv'

#create the directory if it does not exist
os.makedirs(path_to_final_tabtrans, exist_ok=True)

sample_sizes = [10] # 100, 80, 60, 40, 20, 10

for sample in sample_sizes:
    path_of_size = f'{path_to_hyperparameters}/{sample}'

    if sample == 100:
        
        #Import the data
        X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

        path_of_results = f'{path_of_size}/results.csv'

        hyperparameters = data.import_hyperparameters(path_of_results, cv = True)


        n_layers = int(hyperparameters["n_layers"])
        n_heads = int(hyperparameters["n_heads"])
        embedding_size = int(hyperparameters["embedding_size"])
        batch_size = int(hyperparameters["batch_size"])
        epochs = int(hyperparameters["max_epochs_mean"])

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
    
        print(f"The final model for {sample} had finished")


    else:
        path_of_trials = f'{path_to_hyperparameters}/{sample}/trials'

        seed = 11  # You can choose any integer value as the seed
        np.random.seed(seed)

        # Generate 10 random integers between a specified range (e.g., 0 to 100)
        random_seeds = np.random.randint(0, 100, size=10)

        number_of_seed = 1

        for random_seed in random_seeds:

            X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)
            
            general_n_percent = data.reduce_size(y_train, train_indices, val_indices, sample, random_seed)

            #create the new X_train and y_train with the reduced size, this sets are the ones that will enter to the 5 fold cross validation
            X_train = X_train[general_n_percent]
            y_train = y_train[general_n_percent]

            path_of_results = f'{path_of_trials}/seed_{number_of_seed}/results.csv'

            hyperparameters = data.import_hyperparameters(path_of_results, cv = True)

            n_layers = int(hyperparameters["n_layers"])
            n_heads = int(hyperparameters["n_heads"])
            embedding_size = int(hyperparameters["embedding_size"])
            batch_size = int(hyperparameters["batch_size"])
            epochs = int(hyperparameters["max_epochs_mean"])

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

        # List to hold the data from all results.csv files
        all_data = []

        # Iterate over each subfolder in final_tab_size_path
        for subfolder in os.listdir(final_tab_size_path):
            subfolder_path = os.path.join(final_tab_size_path, subfolder)

            results_file = os.path.join(subfolder_path, 'results.csv')

            df = pd.read_csv(results_file)
            all_data.append(df)

            combined_results = pd.concat(all_data, ignore_index=True)
        
        # Now calculate the mean and std for columns 1 to 6 [balanced_accuracy, accuracy, f1, precision, recall, time_trainning]
        mean_values = combined_results.iloc[:, :6].mean()
        std_values = combined_results.iloc[:, :6].std()

        # Create a dictionary with the new column names
        mean_columns = {col: f"{col}_mean" for col in combined_results.columns[:8]}
        std_columns = {col: f"{col}_std" for col in combined_results.columns[:8]}

        # Create a new DataFrame with the results, renaming columns accordingly
        summary_df = pd.DataFrame({
            **{mean_columns[col]: [mean_values[col]] for col in combined_results.columns[:8]},
            **{std_columns[col]: [std_values[col]] for col in combined_results.columns[:8]}
        })

        # Optionally, you can reset the index to avoid having index numbers in the final result
        summary_df.reset_index(drop=True, inplace=True)

        path_for_general_results = os.path.join(f'{path_to_final_tabtrans}/{sample}', 'results.csv')

        summary_df.to_csv(path_for_general_results, index=False)


        print(f"All Seeds have been trained for sample size:{sample}")