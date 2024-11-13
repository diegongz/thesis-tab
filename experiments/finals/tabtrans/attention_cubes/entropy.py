'''
In this file we will import the models that were saved for certain epochs and extract the attention cubes.
'''
import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of this file

# Get the project directory
project_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_folder)))) #parent folder of current_folder

#print(project_path)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files
from utils import attention_file, data
import numpy as np
from pathlib import Path
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
import time



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

df_id = 2
sample_size = 100

for df_id in [1484, 31, 12, 9964, 233092, 3485, 41966]:
    #Let's extract the epochs
    name_df = data.get_dataset_name(df_id)

    chekpoints_path = f'{project_path}/Final_models_4/{name_df}/tabtrans/final_tabtrans_cv/{sample_size}/checkpoints'

    epochs = [] #here all the epochs numebrs will be saved

    for subfolder in Path(chekpoints_path).iterdir():
        epoch_name = os.path.basename(subfolder)
        epoch_number = int(epoch_name[6:])  # Slicing from index 6 to the end
        epochs.append(epoch_number)

    epochs.sort()

    epoch_average_entropy = []

    #define the parameters for compute the variational distance with the uniform distribution
    epoch_average_variational_distance = []


    # Assuming epochs is a list of epoch numbers
    # Define the grid size (rows and columns) for subplots
    num_epochs = len(epochs)
    rows = 3  # Adjust columns based on preference
    cols = 2  # Adjust to limit columns in layout

    # Create a figure and subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten() 

    for i, epoch in enumerate(epochs):
        print(f'Extracting attention matrix for epoch {epoch}')
        
        attention_matrix = attention_file.matrix_for_epoch(df_id, epoch, sample_size, project_path)
        
        n_features = attention_matrix.shape[1]
        
        #Create vector with uniform distribution of len n_features
        uniform_distribution = np.ones(n_features) / n_features
        
        var_distances_rows = []
        #Compute the variational distance
        for row in attention_matrix:
            variational_distance_row = attention_file.variation_distance(row, uniform_distribution)
            var_distances_rows.append(variational_distance_row)
        
        #compute the mean of the variational distance
        mean_variational_distance = np.mean(var_distances_rows)
        epoch_average_variational_distance.append(mean_variational_distance)
            
        ''' 
        Calculate the entropy of each row of the attention matrix
        For this vectorization will be used (using numpy) in order to speed up the process
        axis 1 refers to the rows

        Compute the entropy for each row and normalize by max entropy (log2(n_features)) 
        in order to have values between 0 and 1
        '''
        entropy_per_row = np.apply_along_axis(entropy, 1, attention_matrix, base=2) / np.log2(n_features)

        # Plot on the i-th axis
        axes[i].hist(entropy_per_row, color='navy', edgecolor='black', alpha=0.7)
        axes[i].set_xlabel('Entropy')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'Epoch {epoch}')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)

            #Disable scientific notation on x-axis
        axes[i].xaxis.set_major_locator(MaxNLocator())  # Proper number of ticks
        axes[i].xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.6f}'))  # Show 6 decimals
        
        # Calculate the average of the normalized entropy values
        average_entropy = np.mean(entropy_per_row)
        axes[i].axvline(average_entropy, color='red', linestyle='--', linewidth=1.5, label=f'Median: {average_entropy:.2f}')
        
        epoch_average_entropy.append(average_entropy)

    #--------------------------------------------------------------------------------------------
    # Create a heatmap of the attention matrix
    os.makedirs(f'{project_path}/Final_models_4/{name_df}/comparison', exist_ok=True)
    path_to_save = f'{project_path}/Final_models_4/{name_df}/comparison/heatmap_{name_df}.png'

    attention_file.heatmap_matrix(attention_matrix, name_df, path_to_save)
    #--------------------------------------------------------------------------------------------

    #Histogram of the entropy distribution

    # Adjust layout and show
    fig.suptitle(f'Entropy Distribution Across Epochs {name_df}', fontsize=16)
    # Hide any unused subplots if there are any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout to make space for the main title and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust rect to give space for suptitle


    path_to_save = f'{project_path}/Final_models_4/{name_df}/comparison/entropy_histogram_{name_df}.png'
    plt.savefig(path_to_save,  dpi=300)
    #--------------------------------------------------------------------------------------------

    #plot x vs epoch_average_entropy
            
    #plot x vs epoch_average_entropy
    plt.figure(figsize=(15, 6))

    # Plot sample_size vs xgboost_means_ba
    plt.plot(epochs, epoch_average_entropy, 
            marker='s', markersize=8, color='navy', 
            linewidth=2, linestyle='-')

    # Enhancing aesthetics
    plt.title(f'Entropy through Epochs {name_df}', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Average Entropy', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)  # Add grid for better readability
    plt.legend(fontsize=12)

    # To avoid scientific notation and display values properly
    plt.gca().yaxis.set_major_locator(MaxNLocator())  # Ensures a proper number of ticks
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.6f}'))  # Show 6 decimals

    plt.tight_layout()

    path_to_save = f'{project_path}/Final_models_4/{name_df}/comparison/entropy_norm_plot_{name_df}.png'
    plt.savefig(path_to_save,  dpi=300)
    #--------------------------------------------------------------------------------------------

    #plot x vs epoch_average_variational_distance
            
    #plot x vs epoch_average_entropy
    plt.figure(figsize=(15, 6))

    # Plot sample_size vs xgboost_means_ba
    plt.plot(epochs, epoch_average_variational_distance, 
            marker='s', markersize=8, color='navy', 
            linewidth=2, linestyle='-')

    # Enhancing aesthetics
    plt.title(f'Variational Distance through Epochs {name_df}', fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('Average Variational Distance', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(alpha=0.3)  # Add grid for better readability
    plt.legend(fontsize=12)

    # To avoid scientific notation and display values properly
    plt.gca().yaxis.set_major_locator(MaxNLocator())  # Ensures a proper number of ticks
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.6f}'))  # Show 6 decimals

    plt.tight_layout()

    path_to_save = f'{project_path}/Final_models_4/{name_df}/comparison/variational_distance_plot_{name_df}.png'
    plt.savefig(path_to_save,  dpi=300)
