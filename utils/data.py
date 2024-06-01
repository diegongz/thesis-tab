import numpy as np
import os
from sklearn import datasets, model_selection, pipeline, metrics
import pandas as pd
from sklearn.impute import KNNImputer
import openml
import json 
from config import DATA_BASE_DIR
from . import log
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler #to create one hot encoding for categorical variables
import matplotlib.pyplot as plt
import statistics

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

def plot_distribution(x):
    # Create a histogram
    plt.hist(x, bins=np.arange(min(x), max(x) + 1.5) - 0.5, edgecolor='black', alpha=0.7)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Values Distribution')

    # Show the plot
    plt.show()



def import_data(id): #we want to use the task id

    task_id = id
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id #suppose we input the task id 
    df = read_dataset_by_id(dataset_id)

    X = df["features"] #features
    y = df["outputs"].codes #outputs
    
    categorical_features = df['categorical'].tolist() #name of the categorical features
    numerical_features = df['numerical'].tolist() #name of the numerical features

    # Split the data into training and testing sets
    seed = 11
    #the "_prev" is because I will use that set to split again and obtain validaiton and train
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state= seed, stratify=y)
    train_indices, val_indices = model_selection.train_test_split(np.arange(X_train.shape[0]), test_size=1/3, random_state= seed, stratify=y_train) #1/3 of train is equal to 20% of total

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
    n_labels = len(df["labels"].keys()) #number of labels


    ''' 
    print("Distribution of y")
    plot_distribution(y)

    print("Distribution of y_train")
    y_train_final = y_train[train_indices_return]
    plot_distribution(y_train_final)

    print("Distribution of y_validation")
    y_val_final = y_train[val_indices_return]
    plot_distribution(y_val_final)
    '''

    return X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories


def get_dataset_name(task_id):
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)
    return dataset.name
       

