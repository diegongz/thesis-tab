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
20 mfeat-pixel 241 #have probelms this dataset check this one
9964 semeion 257 pass
233092 arrhythmia 280
3485 scene 300
9976 madelon 501
3481 isolet 618 (TO MUCH INSTANCES) 
'''


df_id = 1484
sample_size = [100, 80, 60, 40, 20, 10]

name_df = data.get_dataset_name(df_id)

path_of_datset = f'{project_path}/Final_models_4/{name_df}' #The path can be changed

tabtrans_folder = f'{path_of_datset}/tabtrans/final_tabtrans_cv'
xgboost_folder = f'{path_of_datset}/xgboost/final_xgboost'

tabtrans_means_ba = [] #ba = balanced accuracy
tabtrans_std_ba = []

tabtrans_means_a = [] #a = accuracy
tabtrans_std_a = []

xgboost_means_ba = []
xgboost_std_ba = []

xgboost_means_a = []
xgboost_std_a = []


for sample in sample_size:
    result_tabrans = f"{tabtrans_folder}/{sample}/results.csv"
    result_xgboost = f"{xgboost_folder}/{sample}/results.csv"

    df_tabtrans = pd.read_csv(result_tabrans)
    df_xgboost = pd.read_csv(result_xgboost)

    if sample == 100:

        tabtrans_means_ba.append(df_tabtrans['balanced_accuracy'].iloc[0])
        tabtrans_means_a.append(df_tabtrans['accuracy'].iloc[0])
        
        tabtrans_std_ba.append(0)
        tabtrans_std_a.append(0)

        xgboost_means_ba.append(df_xgboost['balanced_accuracy'].iloc[0])
        xgboost_means_a.append(df_xgboost['accuracy'].iloc[0])

        xgboost_std_ba.append(0)
        xgboost_std_a.append(0)
    
    else:
        tabtrans_means_ba.append(df_tabtrans['balanced_accuracy_mean'].iloc[0])
        tabtrans_means_a.append(df_tabtrans['accuracy_mean'].iloc[0])
        
        tabtrans_std_ba.append(df_tabtrans['balanced_accuracy_std'].iloc[0])
        tabtrans_std_a.append(df_tabtrans['accuracy_std'].iloc[0])

        xgboost_means_ba.append(df_xgboost['balanced_accuracy_mean'].iloc[0])
        xgboost_means_a.append(df_xgboost['accuracy_mean'].iloc[0])

        xgboost_std_ba.append(df_xgboost['balanced_accuracy_std'].iloc[0])
        xgboost_std_a.append(df_xgboost['accuracy_std'].iloc[0])


#Plot for Balanced Accuracy
# Create the plot
plt.figure(figsize=(10, 6))

# Plot sample_size vs xgboost_means_ba with error bars for the standard deviation
plt.errorbar(sample_size, xgboost_means_ba, yerr=xgboost_std_ba, 
             fmt='s-', capsize=5, ecolor='#C8102E', elinewidth=1, 
             marker='s', color='red', label='XGBoost Mean ± Std')

# Plot sample_size vs tabtrans_means_ba with error bars for the standard deviation
plt.errorbar(sample_size, tabtrans_means_ba, yerr=tabtrans_std_ba, 
             fmt='o-', capsize=5, ecolor='#003366', elinewidth=1, 
             marker='o', color='blue', label='Tabtrans Mean ± Std')


# Optionally, invert x-axis if needed
plt.gca().invert_xaxis()  # Uncomment this line if you want to reverse the x-axis

# Add labels and title
plt.xlabel('Sample Size Percentage')
plt.ylabel('Balanced Accuracy Means')
plt.title(f'Balanced Accuracy Dataset {name_df}')
plt.legend()



path_of_comparison = f'{path_of_datset}/comparison'
os.makedirs(path_of_comparison, exist_ok=True)

# Save the plot
plt.savefig(f'{path_of_comparison}/plot_balanced_accuracy.png')



# CREATE THE PLOT FOR BALANCED ACCURACY
plt.figure(figsize=(10, 6))

# Plot sample_size vs xgboost_means_ba with error bars for the standard deviation
plt.errorbar(sample_size, xgboost_means_a, yerr=xgboost_std_a, 
             fmt='s-', capsize=5, ecolor='#C8102E', elinewidth=1, 
             marker='s', color='red', label='XGBoost Mean ± Std')

# Plot sample_size vs tabtrans_means_ba with error bars for the standard deviation
plt.errorbar(sample_size, tabtrans_means_a, yerr=tabtrans_std_a, 
             fmt='o-', capsize=5, ecolor='#003366', elinewidth=1, 
             marker='o', color='blue', label='Tabtrans Mean ± Std')


# Optionally, invert x-axis if needed
plt.gca().invert_xaxis()  # Uncomment this line if you want to reverse the x-axis

# Add labels and title
plt.xlabel('Sample Size Percentage')
plt.ylabel('Accuracy Means')
plt.title(f'Accuracy Dataset {name_df}')
plt.legend()

# Save the plot
plt.savefig(f'{path_of_comparison}/plot_accuracy.png')


print('Plots saved successfully')