import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of this file

# Get the project directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(current_folder))) #parent folder of current_folder

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files

from utils import data
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns 
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
datasets_ids = [1484, 31, 12, 9964, 233092, 3485, 41966] #1484, 31, 12, 9964, 233092, 3485, 41966


path_of_models = f"{project_path}/Final_models_4"

#Columns that will be part of the dataframe
f2i_column = []
balanced_accuracy_xgboost_column = []
balanced_accuracy_tabtrans_column = []
datset_column = []
n_categories_column = []
n_instances_column = []
n_features_column = []
sample_percentage_column = []


# Loop through each folder in the directory
for df_id in datasets_ids:
    
    #Get the name of the datasets
    name_df = data.get_dataset_name(df_id)
    
    #Get the total number of features and instances in the dataset
    X_train, _, _, _, _, _, n_instances_train, n_labels, _, _ = data.import_data(df_id)
    
    n_features = X_train.shape[1] - 1 #We delete the target/class column
     
    #Get the path of the dataset
    path_of_datset = f'{path_of_models}/{name_df}'
    
#-------------------------------------------------------------------------------
    #INFO EXTRACTION
    path_of_tabtrans = f'{path_of_datset}/tabtrans/final_tabtrans_cv'

    path_of_xgboost = f'{path_of_datset}/xgboost/final_xgboost'


    # Loop through each folder in the directory
    #I iterate through the sample sizes of xgboost because the sample sizes of tabtrans are the same
    for sample_size in os.listdir(path_of_xgboost):
        if sample_size.isdigit():
            
            #Add the first values of the columns
            datset_column.append(name_df)      
            n_features_column.append(n_features)
            n_categories_column.append(n_labels) #Add the number of classes in that dataset

            
            sample_size = int(sample_size)
            sample_percentage_column.append(sample_size)

            #number of instances after doing the reduciton of the sample size 
            n_instances = int((sample_size/100)*n_instances_train)
            n_instances_column.append(n_instances)
            
            f2i_ratio = n_features/n_instances
            f2i_column.append(f2i_ratio)

            
            #Get the balanced accuracies path of the folder
            
            #dataset path of xgboost results
            result_xgboost = f"{path_of_xgboost}/{sample_size}/results.csv"
            
            #dataset path of tabtrans results
            result_tabtrans = f"{path_of_tabtrans}/{sample_size}/results.csv"
            
            if sample_size == 100:
                metric = 'balanced_accuracy'
            else:
                metric = 'balanced_accuracy_mean'
                            
            xgboost_balanced_accuracy = pd.read_csv(result_xgboost)[metric].iloc[0]
            tabtrans_balanced_accuracy = pd.read_csv(result_tabtrans)[metric].iloc[0]
            
            balanced_accuracy_xgboost_column.append(xgboost_balanced_accuracy)
            balanced_accuracy_tabtrans_column.append(tabtrans_balanced_accuracy)

# Create a DataFrame from the lists
data = pd.DataFrame({
    'f2i': f2i_column,
    'balanced_accuracy_xgboost': balanced_accuracy_xgboost_column,
    'balanced_accuracy_tabtrans': balanced_accuracy_tabtrans_column,
    'n_features': n_features_column,
    'dataset': datset_column,
    'n_categories': n_categories_column,
    'n_instances': n_instances_column,
    'sample_percentage': sample_percentage_column
    })

# Reshape data using pd.melt to create a long format
data_melted = pd.melt(
    data, 
    id_vars=['f2i'], 
    value_vars=['balanced_accuracy_xgboost', 'balanced_accuracy_tabtrans'],
    var_name='Model', 
    value_name='Balanced Accuracy'
)


# Calculate maximum value of f2i
max_f2i = round(data_melted['f2i'].max())
print(type(max_f2i))


#Adjust The intervals of the f2i
# Define bins dynamically, ensuring the last bin exceeds the maximum f2i value
bin_size = 2  # Define the size of each bin in other words the step of the bins
bins = np.arange(0, 10 , bin_size) #Create from 0-10 in steps of 2 (it stop at 8)

bins = np.append(bins, max_f2i)  # Add the last bin the maxiumum value we can reach

labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-2)]  # Generate labels
labels.append(f"8-{max_f2i}")  # Add the last interval

data_melted['f2i_interval'] = pd.cut(data_melted['f2i'], bins=bins, labels=labels)

# Map model names for readability
data_melted['Model'] = data_melted['Model'].replace({
    'balanced_accuracy_xgboost': 'XGBoost',
    'balanced_accuracy_tabtrans': 'TabTrans'
})

path_for_images = f"{path_of_models}/all_datasets"

# Create box plot with f2i intervals
fig = px.box(
    data_melted,
    x='f2i_interval',
    y='Balanced Accuracy',
    color='Model',
    color_discrete_map={'XGBoost': 'red', 'TabTrans': 'blue'},
    category_orders={'f2i_interval': labels},
    points="all"
)

fig.update_xaxes(title_text='Feature-to-Instance Ratio Intervals')
# Center the title
fig.update_layout(title=dict(text="Balanced Accuracy across Feature-to-Instance Ratio", x=0.5))

# Export the figure as a PNG image
fig.write_image(f"{path_for_images}/boxplot_f2i.png", scale=2)