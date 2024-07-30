import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__))

# Get the project directory
folder_path = os.path.dirname(current_folder)
project_path = os.path.dirname(folder_path)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files

from utils import tabtrans_file, data

'''
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

tasks = [1484, 12, 9964, 233092, 3485, 9976] #1484, 12, 9964, 233092, 3485, 9976
sample_size = [20,40,60,80,100] # 20,40,60,80,100


for task in tasks:
    for sample in sample_size:
        tabtrans_file.final_tab_trans(task, sample, project_path)

