import numpy as np
import os
from sklearn import datasets, model_selection
import pandas as pd
from sklearn.impute import KNNImputer
import openml
import json 
from config import DATA_BASE_DIR
from . import log
from sklearn.preprocessing import LabelEncoder #to create one hot encoding for categorical variables


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



def import_data(id): #we want to use the task id

    task_id = id
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id #suppose we input the task id 
    df = read_dataset_by_id(dataset_id)

    X = df["features"] #features
    y = df["outputs"].codes #outputs

    categorical_features = df['categorical'].tolist() #name of the categorical features
    numerical_features = df['numerical'].tolist() #name of the numerical features

    # Create numerical and categorical datasets
    X_categorical = X[categorical_features]  # Categorical features
    X_numerical = X[numerical_features]     # Numerical features


    #Fix missing values
    if X_numerical.isnull().values.any():
        imputer = KNNImputer(n_neighbors=10)
        numerical_imputed = imputer.fit_transform(X_numerical)
        X_numerical = pd.DataFrame(numerical_imputed, columns=X_numerical.columns) # Convert NumPy array back to Pandas DataFrame

    
    # Filter out categorical columns with only one unique value
    redundant_columns = [col for col in X_categorical.columns if X_categorical[col].nunique() <= 1]
    X_categorical = X_categorical.drop(columns=redundant_columns)

    # Recompute categorical features after filtering
    categorical_features = [col for col in categorical_features if col not in redundant_columns]

    # Create a LabelEncoder object
    le = LabelEncoder()
    for col in X_categorical.columns:
        X_categorical[col] = le.fit_transform(X_categorical[col].astype(str))


    X_ordered = pd.concat([X_numerical, X_categorical], axis=1)


    n_instances = X_ordered.shape[0]
    n_numerical = X_numerical.shape[1]
    n_categories = [X_categorical[col].nunique() for col in X_categorical.columns] #list that tells the number of categories for each categorical feature
    n_labels = len(df["labels"].keys()) #number of labels

    seed = 11
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_ordered, y, test_size=0.20, random_state= seed, stratify=y)

    X_train = X_train.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)


    train_indices, val_indices = model_selection.train_test_split(np.arange(X_train.shape[0]), test_size=1/3, stratify=y_train) #1/3 of train is equal to 20% of total


    return X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories
    
    
    ''' 
    for col in categorical_features:
        # Factorize the column
        labels, _ = pd.factorize(X_categorical[col])

        # Convert labels to a categorical array
        categorical_labels = pd.Categorical(labels, categories=_, ordered=False)

        # Assign the categorical values back to the DataFrame
        X_categorical.loc[:, col] = categorical_labels
    
    # Factorize categorical features
    for col in categorical_features:
        X_categorical.loc[:, col], _ = pd.factorize(X_categorical[col])

    for col in X_categorical.columns:
        # Add 1 to each value in the column
        X_categorical[col] = X_categorical[col] + 1
    '''
       

def get_dataset_name(task_id):
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)
    dataset_name = dataset.name
    
    return dataset_name