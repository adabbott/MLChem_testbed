from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

def data():
    data = pd.read_csv("PES.dat") 
    # drop duplicates, toggle on or off
    data = data.drop_duplicates(subset = 'E')
    data = data.values
    X = data[:,0:-1]
    y = data[:,-1]
    # should you scale a smaller range to fit multiple activ functions?
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_X = scaler.fit_transform(X)
    scaled_y = scaler.fit_transform(y.reshape(-1,1))
    return scaled_X, scaled_y
    

def create_model(X, y):
    in_dim = tuple([X.shape[1]])
    out_dim = y.shape[1]
    #valid_set = tuple([X_valid, y_valid])
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_losses = []
    cv_maes = []
    for train, test in kfold.split(X, y):
        model = Sequential()
        model.add(Dense(10, input_shape=in_dim))
        model.add(Activation('softmax'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.add(Dense(out_dim))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
        model.fit(x=X[train],y=y[train],epochs=10,batch_size=5,verbose=0)

        scores = model.evaluate(X[test], y[test], verbose=0)
        cv_losses.append(scores[0]) 
        cv_maes.append(scores[1]) 
    print(cv_losses)
    print(cv_maes)
    print("Average MSE and stdev from scaled energies:")
    print("{} (+/- {})".format(np.mean(cv_losses), np.std(cv_losses)))
    print("Average MAE and stdev from scaled energies:")
    print("{} (+/- {})".format(np.mean(cv_maes), np.std(cv_maes)))

    # now hyperopt loss should be the average of the folds losses
    loss = np.mean(cv_losses)
    mae = np.mean(cv_maes)
    print("Test MSE: ", loss)
    print("Test MAE: ", mae)
    # now with CV, the model we have here is just the last one in the CV, unclear what to do here
    return {'loss': loss, 'status': STATUS_OK}


X, y = data()
result = create_model(X,y)
