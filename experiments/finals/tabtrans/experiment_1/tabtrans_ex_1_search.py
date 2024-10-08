#In this experiment for every sample size less than 100, suppose size P, we will repeat the
#experiment of taking a sample of size P% and train the transformer model with the best hyperparameters 
#and the validation set will be taken from the remaining N-P% instances.

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
12 mfeat-factors 217 pass
20 mfeat-pixel 241 #have probelms this dataset check this one
9964 semeion 257 pass
233092 arrhythmia 280
3485 scene 300
9976 madelon 501
3481 isolet 618 (TO MUCH INSTANCES) 

'''




#define the search space

n_layers_lst = [2, 3, 4, 5] #2, 3, 4, 5
n_heads_lst = [4, 8, 16, 32] #4, 8, 16, 32
embed_dim = [128,256] #The embedding size is set one by one to avoid the out of memory error {128, 256}
batch_size = 32 # 32, 64, 128, 256, 512, 1024
epochs = 100
sample_size = [100,80,60,40,20]


df_id = 1484

name_df = data.get_dataset_name(df_id)

path_of_datset = f'{project_path}/Final_models_3/{name_df}' #The path can be changed


tabtrans_dataset_path = f'{path_of_datset}/tabtrans/hyperparameter_selection'

for sample in sample_size:
    if sample == 100:
        
        #import data for the task
        X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

        
        #In this list I will append the dicts that represent the rows of the csv file
        table_of_results = []
        
        size_path = f'{tabtrans_dataset_path}/{sample}'

        # Create the folder and its subfolders (True means that the folder will be created if it does not exist)
        os.makedirs(size_path, exist_ok=True)

        experiment_num = 1
        #train the model
        for n_layers in n_layers_lst:
            for n_heads in n_heads_lst:
                for embedding_size in embed_dim:
                                        
                    model, metrics = tabtrans_file.general_tabtrans(X_train, X_test, y_train, y_test, train_indices, val_indices, n_labels, n_numerical, n_categories, n_layers, n_heads, embedding_size, batch_size, epochs, hyperparameter_search = True)

                    #create the dict of the actual configuration
                    configuration_dict = data.configuration_dict(n_layers, n_heads, embedding_size, batch_size)
                    configuration_dict["max_epochs"] = len(model.history)
             

                    row_result = {**configuration_dict ,**metrics} #this is a row of the csv file

                    table_of_results.append(row_result)



                    #PLOTS SECTION (ERROR_IN VS OUT & ACCURACY VS EPOCHS)
                    #Save the plot of the model
                    path_model_plot = os.path.join(size_path, 'plots')

                    configuration_path = os.path.join(path_model_plot, f'ex_{experiment_num}{n_layers}_{n_heads}_{embedding_size}')
                    
                    #create the folders if they dont exist
                    os.makedirs(configuration_path, exist_ok=True)

                    #create the plot
                    fig = plots.model_plots(model, f"Experiment {experiment_num}")

                    fig.savefig(os.path.join(configuration_path, 'plots.png'))

                    experiment_num += 1

        #convert the list of dicts to a pandas dataframe
        df = pd.DataFrame(table_of_results)

        # Define the full file path where the results csv will be saved
        file_path = os.path.join(size_path, 'results.csv')

        # Convert the DataFrame to a CSV file
        df.to_csv(file_path, index=False)

        print(f"The HYPERPARAMETER SEARCH of tabtrans with sample size {sample} has finished")
    
    
    
    else:
        # Set the seed for reproducibility
        seed = 11  # You can choose any integer value as the seed
        np.random.seed(seed)

        # Generate 10 random integers between a specified range (e.g., 0 to 100)
        random_seeds = np.random.randint(0, 100, size=10)

        number_of_seed = 1

        for random_seed in random_seeds:

            X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

            seed_folder = f'{tabtrans_dataset_path}/{sample}/trials/seed_{number_of_seed}'
            
            # Create the folder and its subfolders (True means that the folder will be created if it does not exist)
            os.makedirs(seed_folder, exist_ok=True)
          
            train_indices, val_indices = data.reduce_size(y_train, train_indices, val_indices, sample, random_seed)

            #In this list I will append the dicts that represent the rows of the csv file
            table_of_results = []

            experiment_num = 1
            #train the model
            for n_layers in n_layers_lst:
                for n_heads in n_heads_lst:
                    for embedding_size in embed_dim:
                                                
                        model, metrics = tabtrans_file.general_tabtrans(X_train, X_test, y_train, y_test, train_indices, val_indices, n_labels, n_numerical, n_categories, n_layers, n_heads, embedding_size, batch_size, epochs, hyperparameter_search = True)
                        
                        #create the dict of the actual configuration
                        configuration_dict = data.configuration_dict(n_layers, n_heads, embedding_size, batch_size)
                        configuration_dict["max_epochs"] = len(model.history)

                        row_result = {**configuration_dict ,**metrics} #this is a row of the csv file

                        table_of_results.append(row_result)

                        #PLOTS SECTION (ERROR_IN VS OUT & ACCURACY VS EPOCHS)
                        #Save the plot of the model
                        path_model_plot = os.path.join(seed_folder, 'plots')

                        configuration_path = os.path.join(path_model_plot, f'ex_{experiment_num}_{n_layers}_{n_heads}_{embedding_size}')
                        
                        #create the folders if they dont exist
                        os.makedirs(configuration_path, exist_ok=True)

                        #create the plot
                        fig = plots.model_plots(model, f"Experiment {experiment_num}")

                        fig.savefig(os.path.join(configuration_path, 'plots.png'))

                        experiment_num += 1
            
            #convert the list of dicts to a pandas dataframe
            df = pd.DataFrame(table_of_results)

            # Define the full file path where the results csv will be saved
            file_path = os.path.join(seed_folder, 'results.csv')

            # Convert the DataFrame to a CSV file
            df.to_csv(file_path, index=False)

            print(f"The HYPERPARAMETER SEARCH of tabtrans with sample size {sample} and seed {random_seed} has finished")

            number_of_seed += 1