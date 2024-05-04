import numpy as np
import os
import pandas as pd
import openml
import json 
from config import DATA_BASE_DIR
from . import log

logger = log.get_logger()

def read_dataset_by_id(
    id
    ):

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


def preprocess(df): #df is the dict from read_dataset_by_id
    
    df_pandas = df["features"]

    categorical_features = df['categorical'].tolist()
    n_categorical = len(categorical_features)

    numerical_features = df['numerical'].tolist()
    n_numerical = len(numerical_features)

    numerical_features_df = df_pandas[numerical_features]  # Assuming numerical_features is a list of column names
    categorical_features_df = df_pandas[categorical_features]  # Assuming categorical_features is a list of column names

    df_ordered = pd.concat([numerical_features_df,categorical_features_df], axis=1) #ordered columns, first numerical then categorical

    #this for loop creates a one-hot encoding for each categorical feature
    for col in categorical_features:
        df_ordered[col], _ = pd.factorize(df_ordered[col])
    
    # Find redundant numerical columns
    redundant_columns_numerical = []
    for column in numerical_features:
        if df_ordered[column].nunique() == 1:  # Check if the column has only one unique value
            redundant_columns_numerical.append(column)
    
    n_numerical = n_numerical - len(redundant_columns_numerical)

    # Find redundant categorical columns
    redundant_columns_categorical = []
    for column in categorical_features:
        if df_ordered[column].nunique() == 1:  # Check if the column has only one unique value
            redundant_columns_categorical.append(column)
    
    n_categorical = n_categorical - len(redundant_columns_categorical)


    # Drop redundant numerical columns
    df_ordered.drop(redundant_columns_numerical, axis=1, inplace=True)

    # Drop redundant categorical columns
    df_ordered.drop(redundant_columns_categorical, axis=1, inplace=True)

    X = df_ordered.values
    y = df["outputs"].codes
    
    """
    Dataset metadata definition.

        n_instances: Number of instances (rows) in your dataset.
        n_numerical: Number of numerical features in your dataset.
        n_categorical: List of the number of categories for each categorical column.
        n_labels: Number of classification labels.
        
    """

    n_instances = X.shape[0]
    n_labels = len(df["labels"].keys())

    return X, y, n_instances, n_numerical, n_categorical, n_labels    #returns the features dataset as a numpy array and the target as a numpy array and meta data
