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
from collections import Counter

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

def plot_distribution(x, ax):
    # Create a histogram on the provided Axes object
    ax.hist(x, bins=np.arange(min(x), max(x) + 1.5) - 0.5, edgecolor='black', alpha=0.7)
    # Add labels and title
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title('Values Distribution')



def import_data(id, sample_percentage): #we want to use the task id

    if id in [1484,1564]: #two selected datasets with no task id, just id
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
    X = X[mask]
    y = y[mask]
    
    unique_values, counts = np.unique(y, return_counts=True)
    n_labels = len(unique_values)

    #create label encoder for the outputs just to have an order in values
    encoder = LabelEncoder()

    #encode the labels (converts strings to integers from 0 to n-1)
    y = encoder.fit_transform(y)

    if sample_percentage != 100:
        #size reduction// Used to reduce instances size from 100%-80%-60%-40%-20%
        X, _, y, _ = model_selection.train_test_split(X, y, train_size = sample_percentage/100, stratify=y, random_state=11)
    
    categorical_features = df['categorical'].tolist() #name of the categorical features
    numerical_features = df['numerical'].tolist() #name of the numerical features

    # Split the data into training and testing sets
    seed = 11

    #the "_prev" is because I will use that set to split again and obtain validaiton and train
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state= seed, stratify = y)
    train_indices, val_indices = model_selection.train_test_split(np.arange(X_train.shape[0]), test_size=1/3, random_state= seed, stratify = y_train) #1/3 of train is equal to 20% of total

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

#-----------------------------------------------------------------------------------------------------
# Plot the distributions of y, y_train, and y_validation
    '''
    # Create a figure with 1 row and 3 columns
    fig, axs = plt.subplots(1, 3, figsize=(18, 5)) # figsize adjusts the size of the figure

    # Plot the distributions
    plot_distribution(y, axs[0])
    axs[0].set_title('Distribution of y')

    plot_distribution(y_train_final, axs[1])
    axs[1].set_title('Distribution of y_train')

    plot_distribution(y_val_final, axs[2])
    axs[2].set_title('Distribution of y_validation')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Display the plots
    plt.show()
    '''

    return X_train, X_test, y_train, y_test, train_indices, val_indices, n_instances, n_labels, n_numerical, n_categories


def get_dataset_name(task_id):
    task = openml.tasks.get_task(task_id)
    dataset_id = task.dataset_id
    dataset = openml.datasets.get_dataset(dataset_id)
    return dataset.name
       

