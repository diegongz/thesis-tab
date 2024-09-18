import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__)) #path of the folder that contains this file

# Get the project directory
folder_path = os.path.dirname(current_folder) #parent folder of current_folder
project_path = folder_path #In this case works

#project_path = os.path.dirname(folder_path)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files

from utils import fast_model

#task_id = 233090 #anneal
#task_id = 233093 #mfeat
#task_id = 233092 #arrhythmia
#task_id = 233108 #cnae-9
#task_id = 233118 #fashion MNIST
''' 
233118 fashion-mnist 784 features
233133 falbert 801 features
233108 cnae-9 857 features
233121 Devnagari-Script 1025 features
233131 christine 1637 features 
233132 dilbert 2001 features


FINAL DATASETS
12 mfeat-factors 217 pass
20 mfeat-pixel 241 #have probelms this dataset check this one
9964 semeion 257 pass
233092 arrhythmia 280
3485 scene 300
9976 madelon 501
3481 isolet 618 (TO MUCH INSTANCES) 


---------Datasets with id not task id----------
1484 lsvt
1564 dbworld-subjects
'''

tasks = [233092, 3485] #1484, 12, 9964, 233092, 3485, 9976

n_layers_lst = [2, 3, 4, 5] #2, 3, 4, 5
n_heads_lst = [4, 8, 16, 32] #4, 8, 16, 32
embed_dim = [128,256] #The embedding size is set one by one to avoid the out of memory error {128, 256}
batch_size = 32 # 32, 64, 128, 256, 512, 1024
epochs = 100
sample_size = [100,80,60,40,20]

for task_id in tasks:
    fast_model.train_model(task_id, sample_size, n_layers_lst, n_heads_lst, embed_dim, batch_size, epochs, project_path)

