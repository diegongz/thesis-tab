import sys
import os

# Get the directory of the current script
current_folder = os.path.dirname(os.path.abspath(__file__))

# Get the project directory
folder_path = os.path.dirname(current_folder)
project_path = os.path.dirname(folder_path)

sys.path.append(project_path) #This helps to be able to import the data from the parent directory to other files

from utils import tabtrans_file, data


tasks = [12, 9964] #1484, 12, 9964, 233092, 3485, 9976
#sample_size = [20,40,60,80,100] # 20,40,60,80,100

for ds_id in tasks:
    data.compare_models(ds_id, project_path)