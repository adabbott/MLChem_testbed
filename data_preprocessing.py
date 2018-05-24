import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
import os



def scale_data(datapath):
    data = pd.read_csv(datapath)
    ncols = data.shape[1]
    # assume only last column is the target
    nfeatures = ncols - 1
    print(nfeatures)
    init_x = data.iloc[:, :nfeatures].values
    print(len(init_x))
    init_y = data.iloc[:, -1].values 

    #scale the data (choice of feature_range may need to be overridden for some activation functions)
    scaler = MinMaxScaler(feature_range=(-1,1))
    X = np.empty(shape=(len(init_x),nfeatures))
    for i, col in enumerate(init_x.T):
        X[:,i] = np.squeeze(scaler.fit_transform(col.reshape(-1,1)))

    y = scaler.fit_transform((init_y).reshape(-1,1))
    return X, y

def split_data(train_percent, valid_percent, test_percent):
    """
    Split data into training, validation, and testing sets.
    Parameters
    ----------
    train_percent : float 
        Percentage of the total dataset for training 
    valid_percent : float 
        Percentage of the total dataset for validation
    test_percent : float 
        Percentage of the total dataset for testing 
    """
    total = train_percent + valid_percent + test_percent
    if (total < .99) or (total > 1.01):
        raise Exception("Data division definitions do not add up to 1")

    train_split = 1 - train_percent
    test_val_split =  test_percent / (valid_percent + test_percent)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size = train_split, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp, test_size = test_val_split , random_state=42)

    train_set = tuple([X_train, y_train])
    valid_set = tuple([X_valid, y_valid])
    test_set = tuple([Xtest, y_test]) 

    return train_set, valid_set, test_set

def split_data_CV():
    pass


scale_data("PES.dat")
