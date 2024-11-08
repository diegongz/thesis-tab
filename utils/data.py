import numpy as np
import os
from sklearn import datasets, model_selection, pipeline, metrics
import pandas as pd
from sklearn.impute import KNNImputer
import openml
import json 
#from config import DATA_BASE_DIR
#from . import log
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler #to create one hot encoding for categorical variables
import matplotlib.pyplot as plt
from collections import Counter

#logger = log.get_logger()

def read_dataset_by_id(id):

    dataset_info = openml.datasets.get_dataset(id, download_data=False)
    target = dataset_info.default_target_attribute

    features, outputs, categorical_mask, columns = dataset_info.get_data(
            dataset_format="dataframe", target=target
        )

    # Remove rows with all nans
    features = features.dropna(axis=0, how="all")
    # Remove columns with all nans
    features = features.dropna(axis=1, how="all")

    removed_cols = set(columns) - set(columns).intersection(set(features.columns))

    removed_mask = np.isin(columns, list(removed_cols))
    columns = np.array(columns)
    columns = columns[~removed_mask]
    
    categorical_mask = np.array(categorical_mask)
    categorical_mask = categorical_mask[~removed_mask]

    assert features.shape[0] == outputs.shape[0], "Invalid features and predictions shapes"

    labels = {value: idx for idx, value in enumerate(outputs.unique().categories.values)}
    outputs = outputs.cat.rename_categories(labels).values

    categorical = columns[categorical_mask]
    numerical = columns[~categorical_mask]

    categories = {}

    for col in categorical:
        categories[col] = features[col].dropna().unique().categories.values

    return {
        "features": features,
        "outputs": outputs,
        "target": target,
        "labels": labels,
        "columns": columns,
        "categorical": categorical,
        "categories": categories,
        "n_categorical": [ len(categories[k] )for k in categories ],
        "numerical": numerical,
        "n_numerical": len(numerical)
    }

#def read_meta_csv(dirname, file_prefix):
    dataset_file = os.path.join(dirname, f"{file_prefix}.csv")
    meta_file = os.path.join(dirname, f"{file_prefix}.meta.json")
    data = pd.read_csv(dataset_file)

    with open(meta_file, "r") as f:
        meta = json.load(f)

    return data, meta

def read_dataset(dataset):

    datasets_dirname = os.path.join(DATA_BASE_DIR, dataset)
    train_data, dataset_meta = read_meta_csv(datasets_dirname, "train")
    train_indices = dataset_meta["df_indices"]
    logger.info(f"Training size: {train_data.shape}")

    test_data, test_dataset_meta = read_meta_csv(datasets_dirname, "test")
    test_indices = test_dataset_meta["df_indices"]
    logger.info(f"Test size: {test_data.shape}")

    data = pd.concat([train_data, test_data], axis=0)
    logger.info(f"Total size: {data.shape}")

    logger.info("Sorting dataset as original")
    indices = train_indices + test_indices
    data.index = indices
    data = data.sort_index()
    
    return data, dataset_meta

#We take a numeric dataset, we will compute mean and std for each column and we will normalize the data
def mean_and_std(df):
    #Calculate mean and standard deviation
    mean = df.mean()
    std = df.std()

    return mean, std

def normalize(df, mean, std):
    # Avoid division by zero by replacing zero std with 1 (no scaling for these columns)
    std_replaced = std.replace(0, 1)
    
    # Normalize the data
    df_normalized = (df - mean) / std_replaced

    return df_normalized

def plot_distribution(x, ax):
    # Create a histogram on the provided Axes object
    ax.hist(x, bins=np.arange(min(x), max(x) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
    # Add labels and title
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Values Distribution')


#This function helps to get faster the portion of the trainning set that will be used for validation to use in the import data function
def fraction_from_trainning(trainning_per, validation_per):
    dec_trainning = trainning_per/100
    dec_validation = validation_per/100

    portion_from_trainning = dec_validation/dec_trainning

    return portion_from_trainning

def configuration_dict(n_layers, n_heads, embedding_size, batch_size):
    
    parameters = {
    "n_layers": n_layers,
    "n_heads": n_heads,
    "embedding_size": embedding_size,
    "batch_size": batch_size,
    }

    return parameters

#-------------------------------------------------------------------------------------------
'''
This function will import the data from the openml dataset, it will preprocess 
the data and return the train and test 
Considering the whole dataset using a 80-10-10 split
The seed used is 11
'''
def import_data(id): #we want to use the task id

    #set seed
    seed = 11

    if id in [1484,1564,41966,2]: #two selected datasets with no task id, just id
        df = read_dataset_by_id(id)
    else:
        task_id = id
        task = openml.tasks.get_task(task_id)
        dataset_id = task.dataset_id #suppose we input the task id 
        df = read_dataset_by_id(dataset_id)

    X = df["features"] #features
    y = df["outputs"].codes #outputs

    label_counts = Counter(y) #counts how many instances of each class I have

    labels_to_keep = {label for label, count in label_counts.items() if count > 10} #I will keep only the classes that have more than 25 instances

    mask = np.isin(y, list(labels_to_keep))
    
    #Filtering the classes with less than 10 instances
    X_masked = X[mask]
    y_masked = y[mask]
    
    unique_values, counts = np.unique(y_masked, return_counts=True)
    n_labels = len(unique_values)

    #create label encoder for the outputs just to have an order in values
    encoder = LabelEncoder()

    #encode the labels (converts strings to integers from 0 to n-1)
    y_masked = encoder.fit_transform(y_masked)

    #First I want to get the 80-10-10 split (Train-Test-Validation)
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_masked, y_masked, test_size=0.20, random_state= seed, stratify = y_masked)
    
    val_test_size = int(X_train.shape[0]*(.20))

    train_indices, val_indices = model_selection.train_test_split(np.arange(X_train.shape[0]), test_size = val_test_size, random_state= seed, stratify = y_train) #.33 of train is equal to 20% of total

    #The X_test and y_test will be the same for all sample size, what will change is the X_train and y_train
    #If there exists a reduction it will be done in the indices of train indices and val indices
    '''' 
    if sample_percentage != 100:
        #size reduction// Used to reduce instances size from 100%-80%-60%-40%-20% on the training set
        train_indices, _ = model_selection.train_test_split(train_indices, train_size = sample_percentage/100, random_state= 11, stratify = y_train[train_indices]) #.33 of train is equal to 20% of total
    '''

    categorical_features = df['categorical'].tolist() #name of the categorical features
    numerical_features = df['numerical'].tolist() #name of the numerical features

    # Split the data into training and testing sets
    seed = 11

    X_categorical = X_train[categorical_features]  # Categorical features
    X_numerical = X_train[numerical_features]     # Numerical features

    X_categorical_test = X_test[categorical_features]  # Categorical features
    X_numerical_test = X_test[numerical_features]     # Numerical features

    # Always processing using the imputer. If there were not nan, nothing will happen
    imputer = pipeline.Pipeline([('imputer', KNNImputer(n_neighbors=10)), ('scaler', StandardScaler())])
    imputer = imputer.fit(X_numerical.iloc[train_indices]) #train the imputer with the training data
    numerical_imputed = imputer.transform(X_numerical) #use the imputer for all the X_numerical
    X_numerical = pd.DataFrame(numerical_imputed, columns=X_numerical.columns) # Convert NumPy array back to Pandas DataFrame
    
    numerical_imputed_test = imputer.transform(X_numerical_test) #use the imputer for all the X_numerical_test
    X_numerical_test = pd.DataFrame(numerical_imputed_test, columns=X_numerical_test.columns) # Convert NumPy array back to Pandas DataFrame

    # Use ordinal encoder, not label encoder
    # The nan values and non-existing categories are mapped to -1
    le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    le = le.fit(X_categorical.iloc[train_indices])
    # Adding a 1 ensures that -1->0, 0->1, 1->2 indexing correctly the architecture's embeddings table
    categorical_imputed = le.transform(X_categorical) + 1
    X_categorical = pd.DataFrame(categorical_imputed, columns=X_categorical.columns)

    categorical_imputed_test = le.transform(X_categorical_test) + 1
    X_categorical_test = pd.DataFrame(categorical_imputed_test, columns=X_categorical_test.columns)

    X_ordered = pd.concat([X_numerical, X_categorical], axis=1)
    X_ordered_test = pd.concat([X_numerical_test, X_categorical_test], axis=1)

    X_train = X_ordered.values
    X_test = X_ordered_test.values

    n_instances = X_ordered.shape[0]
    n_numerical = X_numerical.shape[1]
    n_categories = [X_categorical[col].nunique() for col in X_categorical.columns] #list that tells the number of categories for each categorical feature
    #n_labels = len(df["labels"].keys()) #number of labels// its defined above

    # Assuming y, y_train, train_indices_return, and val_indices_return are defined
    y_train_final = y_train[train_indices]
    y_val_final = y_train[val_indices]

    return X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories
    #n_instances: is the number of instances in trainning set
    #n_labels: is the number of classes in the dataset
#-------------------------------------------------------------------------------------------
''' 
Takes random sample from X_train of size 80-60-40-20% of the original size
and then the validation indices will be taken from the complement instances of the training indices
''' 
def reduce_size(y_train, train_indices, val_indices, sample_size, seed):
    
    indices = np.sort(np.append(train_indices,val_indices))

    general_n_percent, _ = model_selection.train_test_split(indices, train_size = sample_size/100, random_state= seed, stratify = y_train)

    return general_n_percent

#-------------------------------------------------------------------------------------------
def get_dataset_name(ds_id):
    if ds_id in [1484,1564,41966,2]: #two selected datasets with no task id, just id
        dataset = openml.datasets.get_dataset(ds_id)
        dataset_name = dataset.name
    else:
        task = openml.tasks.get_task(ds_id)
        dataset_id = task.dataset_id
        dataset = openml.datasets.get_dataset(dataset_id)
        dataset_name = dataset.name

    return dataset_name

'''
This function will import all the results from all the folds and reduce all the info in a single row

        'balanced_accuracy': balanced_accuracy,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,

Note:
The rows of the df should be:
['n_layers', 'n_heads', 'embedding_size', 'batch_size', 'max_epochs', 'balanced_accuracy', 'accuracy', 'log_loss', 'roc_auc', 'f1', 'precision', 'recall']
'''

#This funcion average the results from the 5 Fold cross validation and returns just one row with the mean 
# and std of the statistical columns
def results_cv(df):
    # Columns that will remain the same (assuming these values are the same for all rows)
    constant_columns = ['n_layers', 'n_heads', 'embedding_size', 'batch_size']

    stats_columns = [
        "balanced_accuracy", "accuracy", "f1", "precision", "recall", "max_epochs"
    ]
    
    # Get the mean and std for the statistical columns
    mean_values = df[stats_columns].mean()
    std_values = df[stats_columns].std()

    # Prepare new columns names for mean and std
    new_columns = []
    new_values = []

    for col in stats_columns:
        new_columns.append(f'{col}_mean')
        new_columns.append(f'{col}_std')
        new_values.append(mean_values[col])
        new_values.append(std_values[col])

    # Combine the constant values and the new calculated values
    final_columns =  new_columns + constant_columns
    final_values =  new_values + list(df[constant_columns].iloc[0])

    # Create the new DataFrame with one row
    resuls_df = pd.DataFrame([final_values], columns=final_columns)

    return resuls_df

def hyperparameter_selection_file_cv(size_path):
    # List to hold the data from all results.csv files
    all_data = []

    # Iterate over each subfolder in size_path
    for subfolder in os.listdir(size_path):
        subfolder_path = os.path.join(size_path, subfolder)
        
        # Check if it is a directory
        if os.path.isdir(subfolder_path):
            # Path to results.csv within the subfolder
            results_file = os.path.join(subfolder_path, 'results.csv')
            
            # Check if results.csv exists
            if os.path.exists(results_file):
                # Read the CSV file
                df = pd.read_csv(results_file)
                
                # Append the data to the list
                all_data.append(df)
            else:
                print(f'results.csv not found in {subfolder_path}')

    # Combine all the DataFrames into one
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
    else:
        print("No data found to combine.")

    
    return combined_df 


def import_hyperparameters(path_of_csv, cv = False):

    results = pd.read_csv(path_of_csv)

    if cv:
        max_row = results.loc[results['balanced_accuracy_mean'].idxmax()]
    else:
        # Find the row where the "balanced_accuracy" column has the maximum value
        max_row = results.loc[results['balanced_accuracy'].idxmax()]

    hyperparameters = max_row.to_dict()
    
    return hyperparameters


def compare_models(ds_id, project_path):
    
    if ds_id in [1484,1564]: #two selected datasets with no task id, just id
        dataset = openml.datasets.get_dataset(ds_id)
        dataset_name = dataset.name
    
    else:
        dataset_name = get_dataset_name(ds_id)

    X_train, X_test, y_train, y_test, train_indices, val_indices, _, n_labels, n_numerical, n_categories = import_data(ds_id, 100)

    #compute the feature to instance ratio
    total_instances = X_train.shape[0]+X_test.shape[0]
    total_features = X_train.shape[1]

    sample_sizes = [20,40,60,80,100]

    f2i = [] #here I will store the feature to instance ratios

    for x in sample_sizes:
        ratio = total_features/(total_instances*(x/100))
        f2i.append(ratio)

    ds_path_final_tabtrans = os.path.join(project_path, "Final_models", f"{dataset_name}", "tabtrans","final_tabtrans")
    ds_path_xgboost = os.path.join(project_path, "Final_models", f"{dataset_name}", "xgboost","hyperparameter_selection")

    balanced_acc_tabtrans = []
    balanced_acc_xgboost = []

    for size in sample_sizes:
        result_sample_tabtrans = os.path.join(ds_path_final_tabtrans, f"{size}", "results.csv")
        result_sample_xgboost = os.path.join(ds_path_xgboost, f"{size}", "results.csv")

        tabtrans_result_df = pd.read_csv(result_sample_tabtrans)
        xgboost_result_df = pd.read_csv(result_sample_xgboost)

        # Extract the value from the "balanced_accuracy" column
        balanced_accuracy_tabtrans = tabtrans_result_df["balanced_accuracy"].iloc[0]
        balanced_accuracy_xgboost = xgboost_result_df["balanced_accuracy"].iloc[0]

        balanced_acc_tabtrans.append(balanced_accuracy_tabtrans)
        balanced_acc_xgboost.append(balanced_accuracy_xgboost)

    #Lets plot the results
    fig, ax = plt.subplots()
    ax.plot(f2i, balanced_acc_tabtrans, color='blue', label='TabTrans')
    ax.plot(f2i, balanced_acc_xgboost, color='red', label='XGBoost')

    ax.set_title(f'Model comparison in {dataset_name} dataset')
    ax.set_xlabel('f2i Ratio')
    ax.set_ylabel('Balanced Accuracy')
    ax.legend(loc='upper right')

    #path to save the comparison

    path_to_save_plot = os.path.join(project_path, "Final_models", f"{dataset_name}", f"model_comparison_{dataset_name}.png")

    fig.savefig(path_to_save_plot)








        









    

    

      

