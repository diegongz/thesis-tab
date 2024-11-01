import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of this file

# Get the project directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_folder))) #parent folder of current_folder

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files

from utils import data
import pandas as pd
import matplotlib.pyplot as plt

'''
FINAL DATASETS
1484 lsvt
31 credit 
12 mfeat-factors 217 pass
9964 semeion 257 pass
233092 arrhythmia 280
3485 scene 300
41966 isolet (600x618)
9976 madelon 501
20 mfeat-pixel 241 #have probelms this dataset check this one
'''



#id of all the datasets I want to consider in the box plot comparative analysis
datasets_ids = [1484] #1484, 31, 12, 9964, 233092, 3485, 41966


path_of_models = project_path + "/Final_models_4"

#Columns that will be part of the dataframe
f2i_column = []
balanced_accuracy_column = []
model_column = []
datset_column = []
n_categories_column = []
n_instances_column = []
n_features_column = []

# Loop through each folder in the directory
for df_id in datasets_ids:
    
    #Get the name of the datasets
    name_df = data.get_dataset_name(df_id)
    
    #Get the total number of features and instances in the dataset
    X_train, _, _, _, _, _, n_instances_train, n_labels, n_numerical, n_categories = data.import_data(df_id)
    n_features = X_train.shape[1]
    print(n_features)

    
    #Get the path of the dataset
    path_of_datset = f'{path_of_models}/{name_df}'
    
#-------------------------------------------------------------------------------
    #TABTRANS INFO EXTRACTION
    path_of_tabtrans = f'{path_of_datset}/tabtrans'
    
    
    
    
    path_of_xgboost = f'{path_of_datset}/xgboost'
    
    


