import numpy as np
import os
import pandas as pd
import openml
import json 
from config import DATA_BASE_DIR
from . import log

logger = log.get_logger()

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

def read_meta_csv(dirname, file_prefix):
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


#id: is a number
#type: it can be "task" or "ds_id" from OpenML

#If we use the type is task we have as return the original partition of the task:
# return X_train, X_test, y_train, y_test, n_instances, n_labels, n_numerical, n_categorical

#If the type is ds_id as this doesn't have any partition we return the whole dataset:
# return X, y, n_instances, n_labels, n_numerical, n_categorical


def import_data(id, type): #type can be task or id

    if type == "task":
        task = openml.tasks.get_task(id)
        id = task.dataset_id
    
    df = read_dataset_by_id(id)

    X = df["features"]

    categorical_features = df['categorical'].tolist()
    print(categorical_features)
    #n_categorical = len(categorical_features)
    n_categories = df['n_categorical'] #list of number of categories for each categorical feature

    numerical_features = df['numerical'].tolist()
    n_numerical = len(numerical_features)

    X_numerical = X[numerical_features]  # Assuming numerical_features is a list of column names
    X_categorical = X[categorical_features]  # Assuming categorical_features is a list of column names

    X_ordered = pd.concat([X_numerical, X_categorical], axis=1) #ordered columns, first numerical then categorical
    
    '''
    #this for loop creates a one-hot encoding for each categorical feature
    for col in categorical_features:
        X_ordered[col], _ = pd.factorize(X_ordered[col])
    '''
    
    '''
    # Find redundant numerical columns
    redundant_columns_numerical = []
    for column in numerical_features:
        if X_ordered[column].nunique() == 1:  # Check if the column has only one unique value
            redundant_columns_numerical.append(column)
    
    n_numerical = n_numerical - len(redundant_columns_numerical)

    
    # Find redundant categorical columns
    redundant_columns_categorical = []
    for column in categorical_features:
        if X_ordered[column].nunique() == 1:  # Check if the column has only one unique value
            redundant_columns_categorical.append(column)
    
    n_categorical = n_categorical - len(redundant_columns_categorical)
    n_categorical = n_categorical.tolist()


    # Drop redundant numerical columns
    X_ordered.drop(redundant_columns_numerical, axis=1, inplace=True)

    # Drop redundant categorical columns
    X_ordered.drop(redundant_columns_categorical, axis=1, inplace=True)
    '''

    y = df["outputs"].codes

    n_instances = X_ordered.shape[0]
    n_labels = len(df["labels"].keys())

    if type == "task":
        train_indices, test_indices = task.get_train_test_split_indices() #get the indices of the task partition

        X_train = X_ordered.iloc[train_indices].values
        X_test = X_ordered.iloc[test_indices].values

        y_train = y[train_indices]
        y_test = y[test_indices]

        return X_train, X_test, y_train, y_test, n_instances, n_labels, n_numerical, n_categories
    
    else:
        X = X_ordered.values

        return X, y, n_instances, n_labels, n_numerical, n_categories
    

def get_dataset_name(task_id):
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)
    dataset_name = dataset.name
    
    return dataset_name