'''
EXPERIMENT MAKING CROSS VALIDATION

For every size 100-80-60--40-20, and for every seed a cross validation will be made.
The training set will be divided into 80% training and 20% validation. The training 
set will be used to train the model and the validation set will be used to evaluate the model.

If the sample size is less than 100, the I will take randomly the instances used to train the model

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
from sklearn.model_selection import StratifiedKFold



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
41966 isolet (600x618)
'''




#define the search space

n_layers_lst = [2, 3, 4, 5] #2, 3, 4, 5
n_heads_lst = [4, 8, 16, 32] #4, 8, 16, 32
embed_dim = [128, 256] #The embedding size is set one by one to avoid the out of memory error {128, 256}
batch_size = 32 # 32, 64, 128, 256, 512, 1024
epochs = 100
sample_size = [5,10,20]
seed = 11


df_id = 3485

name_df = data.get_dataset_name(df_id)

path_of_datset = f'{project_path}/Final_models_4/{name_df}' #The path can be changed


tabtrans_dataset_path = f'{path_of_datset}/tabtrans/hyperparameter_selection'

for sample in sample_size:
    if sample == 100:
        
        #import data for the task
        X_train, X_test, y_train, y_test, _, _, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)
            
        #Find if I have multiclass in the y's
        if len(np.unique(y_train)) > 2:
            multiclass_val = True
        
        else:
            multiclass_val = False
        
        #In this list I will append the dicts that represent the rows of the csv file
        #table_of_results = []
        
        size_path = f'{tabtrans_dataset_path}/{sample}'

        # Create the folder and its subfolders (True means that the folder will be created if it does not exist)
        os.makedirs(size_path, exist_ok=True)

        
        #train the model

        '''
        n_layers_lst = [2, 3, 4, 5] #2, 3, 4, 5
        n_heads_lst = [4, 8, 16, 32] #4, 8, 16, 32
        embed_dim = [128,256] #The embedding size is set one by one to avoid the out of memory error {128, 256}
        batch_size = 32 # 32, 64, 128, 256, 512, 1024
        epochs = 100
        sample_size = [100,80,60,40,20]                                             
        '''
        experiment_num = 1
        for n_layers in n_layers_lst:
            for n_heads in n_heads_lst:
                for embedding_size in embed_dim:
                    
                    configuration_path = f"{size_path}/config_{experiment_num}_l{n_layers}_h{n_heads}_e{embedding_size}"

                    #create the folder of the condiguration
                    os.makedirs(configuration_path, exist_ok=True)

                    #create the dict of the actual configuration
                    configuration_dict = data.configuration_dict(n_layers, n_heads, embedding_size, batch_size)

                    configuration_fold_comparison = [] #the ith row of this list is the result of the ith fold

                    fold_num = 1

                    # Initialize StratifiedKFold with 5 splits
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = seed)

                    #Obtain the respectives train_indices and val_indices
                    for train_indices,val_indices in skf.split(X_train, y_train):

                        fold_result = []

                        fold_path = f'{configuration_path}/fold_{fold_num}'
                        os.makedirs(fold_path, exist_ok=True)

                        model, metrics = tabtrans_file.general_tabtrans(X_train, X_test, y_train, y_test, train_indices, val_indices, n_labels, n_numerical, n_categories, n_layers, n_heads, embedding_size, batch_size, epochs, hyperparameter_search = True)                        
                        
                        #The epochs saved are the len of the model minus 10 because the model stops after 10 epochs of no improvement
                        if len(model.history) == 100:
                            configuration_dict["max_epochs"] = len(model.history)
                        else:
                            configuration_dict["max_epochs"] = len(model.history)- 10 

                        row_result = {**metrics, **configuration_dict} #this is a row of the csv file

                        configuration_fold_comparison.append(row_result) #For every fold the row result is saved here
                        
                        fold_result.append(row_result)

                        #convert the result of the fold to a pandas dataframe
                        df_fold = pd.DataFrame(fold_result)

                        #result of the fold path
                        fold_result_path = os.path.join(fold_path, 'results.csv')

                        # Convert the DataFrame to a CSV file
                        df_fold.to_csv(fold_result_path, index=False)

                        #PLOTS SECTION (ERROR_IN VS OUT & ACCURACY VS EPOCHS)
                        #Save the plot of the model                        
                        #create the plot
                        fig = plots.model_plots(model, f"Fold {fold_num}")

                        fig.savefig(os.path.join(fold_path, 'plots.png'))

                        fold_num += 1
                    
                    df_congif = pd.DataFrame(configuration_fold_comparison)

                    overall_fold_results = data.results_cv(df_congif)

                    #path to export the overall results of the configuration
                    overall_result_path = os.path.join(configuration_path, 'results.csv')

                    # Convert the DataFrame to a CSV file
                    overall_fold_results.to_csv(overall_result_path, index=False)
                    

                    print(f"The configuration number {experiment_num} has finished")
                    print("------------------------------------------------------------------------------------")

                    experiment_num += 1

        hyperparameter_results_sample = data.hyperparameter_selection_file_cv(size_path)

        # Define the output path
        output_csv = os.path.join(size_path, 'results.csv')
        
        # Export the combined DataFrame to CSV
        hyperparameter_results_sample.to_csv(output_csv, index=False)


        print(f"The HYPERPARAMETER SEARCH of tabtrans with sample size {sample} has finished")
    
    
    
    else:
        # Set the seed for reproducibility
        np.random.seed(seed)

        # Generate 10 random integers between a specified range (e.g., 0 to 100)
        random_seeds = np.random.randint(0, 100, size=10)

        number_of_seed = 1

        #Here the random seed process starts
        for random_seed in random_seeds:

            X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories = data.import_data(df_id)

            #Folder path of the Seed Sample/trials/seed_i
            seed_folder = f'{tabtrans_dataset_path}/{sample}/trials/seed_{number_of_seed}'
            
            # Create the folder and its subfolders (True means that the folder will be created if it does not exist)
            os.makedirs(seed_folder, exist_ok=True)
          
            #create a random sample of instances 
            general_n_percent = data.reduce_size(y_train, train_indices, val_indices, sample, random_seed)
            
            #create the new X_train and y_train with the reduced size, this sets are the ones that will enter to the 5 fold cross validation
            X_train = X_train[general_n_percent]
            y_train = y_train[general_n_percent]

            #In this list I will append the dicts that represent the rows of the csv file
            table_of_results = []

            experiment_num = 1
            #train the model
            for n_layers in n_layers_lst:
                for n_heads in n_heads_lst:
                    for embedding_size in embed_dim:
                        
                        #Path one specific configuration in the seed_i
                        configuration_path = f"{seed_folder}/config_{experiment_num}_l{n_layers}_h{n_heads}_e{embedding_size}"

                        #create the folder of the condiguration
                        os.makedirs(configuration_path, exist_ok=True)

                        #create the dict of the actual configuration
                        configuration_dict = data.configuration_dict(n_layers, n_heads, embedding_size, batch_size)

                        #the ith row of this list is the result of the ith fold
                        configuration_fold_comparison = [] 

                        fold_num = 1

                        # Initialize StratifiedKFold with 5 splits
                        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state = seed)

                        #The stratified kfold will divide the training set into 5 folds 
                        for train_indices,val_indices in skf.split(X_train, y_train):
                            fold_result = []

                            fold_path = f'{configuration_path}/fold_{fold_num}'
                            os.makedirs(fold_path, exist_ok=True)

                            model, metrics = tabtrans_file.general_tabtrans(X_train, X_test, y_train, y_test, train_indices, val_indices, n_labels, n_numerical, n_categories, n_layers, n_heads, embedding_size, batch_size, epochs, hyperparameter_search = True)                        

                            #The epochs saved are the len of the model minus 10 because the model stops after 10 epochs of no improvement
                            if len(model.history) == 100:
                                configuration_dict["max_epochs"] = len(model.history)
                            else:
                                configuration_dict["max_epochs"] = len(model.history)- 10 
                            
                            row_result = {**metrics, **configuration_dict} #this is a row of the csv file

                            #For every fold the row result is saved here
                            configuration_fold_comparison.append(row_result)
                            
                            fold_result.append(row_result)

                            #convert the result of the fold to a pandas dataframe
                            df_fold = pd.DataFrame(fold_result)

                            #result of the fold path
                            fold_result_path = os.path.join(fold_path, 'results.csv')

                            # Convert the DataFrame to a CSV file
                            df_fold.to_csv(fold_result_path, index=False)

                            #PLOTS SECTION (ERROR_IN VS OUT & ACCURACY VS EPOCHS)
                            #Save the plot of the model                        
                            #create the plot
                            fig = plots.model_plots(model, f"Fold {fold_num}")

                            fig.savefig(os.path.join(fold_path, 'plots.png'))

                            fold_num += 1

                        df_congif = pd.DataFrame(configuration_fold_comparison)

                        #For all the folds of the configuration, the results are averaged
                        #This file is the average result of the 5 folds for that specific configuration
                        overall_fold_results = data.results_cv(df_congif)

                        #path to export the overall results of the configuration
                        overall_result_path = os.path.join(configuration_path, 'results.csv')

                        # Convert the DataFrame to a CSV file
                        overall_fold_results.to_csv(overall_result_path, index=False)

                        print(f"The configuration number {experiment_num} has finished for seed {random_seed}")
                        print("------------------------------------------------------------------------------------")

                        experiment_num += 1

            hyperparameter_results_sample = data.hyperparameter_selection_file_cv(seed_folder)

            # Define the output path
            output_csv = os.path.join(seed_folder, 'results.csv')
        
            # Export the combined DataFrame to CSV
            hyperparameter_results_sample.to_csv(output_csv, index=False)


            print(f"The HYPERPARAMETER SEARCH of tabtrans with sample size {sample} and seed {random_seed} has finished")

            number_of_seed += 1