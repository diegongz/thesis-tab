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
from pathlib import Path
from scipy.stats import entropy
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

df_id = 31
sample_size = 100

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

for epoch in epochs:
    print(f'Extracting attention matrix for epoch {epoch}')

    entropy_per_row = [] #Here i will save the entropy of each row of the attention matrix
    
    attention_matrix = attention_file.matrix_for_epoch(df_id, epoch, sample_size, project_path)
    
    for row in attention_matrix:
        row_entropy = entropy(row, base=2)
        entropy_per_row.append(row_entropy)
    
    average_entropy = sum(entropy_per_row) / len(entropy_per_row)
    epoch_average_entropy.append(average_entropy)
    
    
    

#plot x vs epoch_average_entropy
plt.figure(figsize=(10, 6))

# Plot sample_size vs xgboost_means_ba
plt.plot(epochs, epoch_average_entropy, 
         marker='s', markersize=8, color='red', 
         linewidth=2, linestyle='-')

# Enhancing aesthetics
plt.title(f'Entropy through Epochs {name_df}', fontsize=16, fontweight='bold')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Average Entropy', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid(alpha=0.3)  # Add grid for better readability
plt.legend(fontsize=12)
plt.tight_layout()


path_to_save = f'{project_path}/Final_models_4/{name_df}/comparison/entropy_plot.png'
plt.savefig(path_to_save,  dpi=300)



