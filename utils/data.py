import numpy as np
import os
from sklearn import datasets, model_selection
import pandas as pd
from sklearn.impute import KNNImputer
import openml
import json 
from config import DATA_BASE_DIR
from . import log
from sklearn.preprocessing import LabelEncoder, StandardScaler #to create one hot encoding for categorical variables
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
    X_train_prev, X_test_prev, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state= seed, stratify=y)

    indices_X_train_prev = X_train_prev.index.tolist() #indices of X_train 
    test_indices = X_test_prev.index.tolist() #indices of X_test

    train_indices, val_indices = model_selection.train_test_split(indices_X_train_prev, test_size=1/3, stratify=y_train) #1/3 of train is equal to 20% of total

    #train_indices converted to the original indices np.arange(X_train.shape[0])
    #this indices are the ones we have to return such that it works in the tabtrans
    train_indices_return = [X_train_prev.index.get_loc(idx) for idx in train_indices]
    val_indices_return = [X_train_prev.index.get_loc(idx) for idx in val_indices] 
    
    complement_indices = val_indices + test_indices #indices of complemnt of training (validation and test indices) 

    X_train = X_train_prev.loc[train_indices] #training set

    X_train_cat = X_train[categorical_features] #training categorical
    X_train_num = X_train[numerical_features] #training numerical

    X_train_complement =  X.loc[complement_indices] #complement of X_train (validation and test set)

    X_train_complement_cat = X_train_complement[categorical_features] #complement of X_train restricted to categorical
    X_train_complement_num = X_train_complement[numerical_features] #complement of X_train restricted to numerical

    ''' 
    Let's impute the missing values in the numerical data with KNN imputer.
    Using the trainning for the imputer in trainning will be used to impute the Validation and Test set
    After that I will normalize the data
    '''

    imputer = KNNImputer(n_neighbors=10)
    X_train_num_imputed = imputer.fit_transform(X_train_num) #this returns a numpy array
    X_train_complement_num_imputed = imputer.transform(X_train_complement_num)

    print("-----------------------------------------")
    print(f"X_train_num_imputed has missing values: {np.isnan(X_train_num_imputed).any()}")
    print(f"X_train_complement_num_imputed has missing values: {np.isnan(X_train_complement_num_imputed).any()}")

    # Convert NumPy array back to Pandas DataFrame
    X_train_num = pd.DataFrame(X_train_num_imputed, columns=X_train_num.columns, index=X_train_num.index) #turnback to pandas
    X_train_complement_num = pd.DataFrame(X_train_complement_num_imputed, columns=X_train_num.columns, index=X_train_complement_num.index) #turnback to pandas

    #remove redundant columns
    redundant_columns = [col for col in X_train_num.columns if X_train_num[col].nunique() <= 1]
    X_train_num = X_train_num.drop(columns=redundant_columns)
    X_train_complement_num = X_train_complement_num.drop(columns=redundant_columns)

    print("-----------------------------------------")
    print("BAck to pandas dataframes")
    print(f"X_train_num has missing values: {X_train_num.isnull().values.any()}")
    print(f"X_train_complement_num has missing values: {X_train_complement_num.isnull().values.any()}")
    display(X_train_num)

    print("------------------------------------------------------------------------------------ ")
    #normalize the data

    # define standard scaler 
    scaler = StandardScaler() 

    # transform data 
    X_train_num_numpy = scaler.fit_transform(X_train_num) #scaler returns a numpy array
    X_train_complement_num_numpy = scaler.transform(X_train_complement_num) #scaler returns a numpy array

    #Transform numpy array to pandas dataframe
    X_train_num = pd.DataFrame(X_train_num_numpy, columns=X_train_num.columns, index=X_train_num.index) #turnback to pandas
    X_train_complement_num = pd.DataFrame(X_train_complement_num_numpy, columns=X_train_num.columns, index=X_train_complement_num.index) #turnback to pandas

    print("-----------------------------------------")
    print(f"X_train_num has missing values: {X_train_num.isnull().values.any()}")
    print(f"X_train_complement_num has missing values: {X_train_complement_num.isnull().values.any()}")

    #Turn back the Numerical splits
    X_val_num = X_train_complement_num.loc[val_indices] #Numerical validation set
    X_test_num = X_train_complement_num.loc[test_indices] #Numerical test set

    print("-----------------------------------------")
    print(f"X_val_num has missing values: {X_val_num.isnull().values.any()}")
    print(f"X_test_num has missing values: {X_test_num.isnull().values.any()}")


    X_train_num_final = pd.concat([X_train_num, X_val_num], axis=0) #concatenate the numerical data (train and validation)
    
    print("-----------------------------------------")
    print(f"X_train_num_final has missing values: {X_train_num_final.isnull().values.any()}")
    
    '''
    Now I will work with the categorical datasets
    Given that for all missing categorical I just need to add -1 to NA then I can merge Train, Val and test
    '''

    X_cat = pd.concat([X_train_cat, X_train_complement_cat], axis=0) #concatenate the categorical data

    # Filter out categorical columns with only one unique value
    redundant_columns = [col for col in X_cat.columns if X_cat[col].nunique() <= 1]
    X_cat = X_cat.drop(columns=redundant_columns)

    # Recompute categorical features after filtering
    categorical_features = [col for col in categorical_features if col not in redundant_columns]

    # Create a LabelEncoder object
    le = LabelEncoder()
    for col in X_cat.columns:
        X_cat[col] = le.fit_transform(X_cat[col].astype(str))

    #Get the respective splits
    X_train_cat = X_cat.loc[train_indices] #Categorical train set
    X_val_cat = X_cat.loc[val_indices] #Categorical validation set
    X_test_cat = X_cat.loc[test_indices] #Categorical test set


    X_train_cat_final = pd.concat([X_train_cat, X_val_cat], axis=0) #concatenate the categorical data

    print("-----------------------------------------")
    print(f"X_train_cat_final has missing values: {X_train_cat_final.isnull().values.any()}")


    #concatenate the numerical and categorical splits
    X_train_final = pd.concat([X_train_num_final, X_train_cat_final], axis=1)

    print("-----------------------------------------")
    print(f"X_train_final has missing values: {X_train_final.isnull().values.any()}")


    #X_train_final = pd.concat([X_train_num, X_train_cat], axis=1)
    #X_val = pd.concat([X_val_num, X_val_cat], axis=1)
    X_test = pd.concat([X_test_num, X_test_cat], axis=1)
    #X_train = pd.concat([X_train_final, X_val], axis=0)

    n_instances = X.shape[0]
    n_numerical = X_train_num_final.shape[1]
    n_categories = [X_cat[col].nunique() for col in X_cat.columns] #list that tells the number of categories for each categorical feature
    n_labels = len(df["labels"].keys()) #number of labels


    X_train = X_train_final.values.astype(np.float32)
    X_test = X_test.values.astype(np.float32)

    print(f"X_train has missing values: {np.isnan(X_train).any()}")

    print("Distribution of y")
    plot_distribution(y)

    print("Distribution of y_train")
    y_train_final = y_train[train_indices_return]
    plot_distribution(y_train_final)

    print("Distribution of y_validation")
    y_val_final = y_train[val_indices_return]
    plot_distribution(y_val_final)
    

    return X_train, X_test, y_train, y_test, train_indices_return, val_indices_return, n_instances, n_labels, n_numerical, n_categories

       

