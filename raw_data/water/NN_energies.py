from keras.models import Sequential
from keras.layers.core import Dense
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle

import pandas as pd
import numpy as np

import sys
import os
import json

# use MLChem 
sys.path.insert(0, "../../../")
import MLChem

data = pd.read_csv("PES.dat") 
x1 = data.iloc[:, 0].values
x2 = data.iloc[:, 1].values
x3 = data.iloc[:, 2].values
y = data.iloc[:, 3].values

scaler = MinMaxScaler(feature_range=(-1,1))
x1 = scaler.fit_transform((x1).reshape(-1,1)) 
x2 = scaler.fit_transform((x2).reshape(-1,1)) 
x3 = scaler.fit_transform((x3).reshape(-1,1)) 
y = scaler.fit_transform((y).reshape(-1,1)) 

X = np.hstack((x1,x2,x3))

X_train, X_fulltest, y_train, y_fulltest = train_test_split(X, y, test_size = 0.5, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_fulltest, y_fulltest, test_size = 0.5, random_state=42)

in_dim = X_train.shape[1]
out_dim = y_train.shape[1]
valid_set = tuple([X_valid, y_valid])


# train a fresh model 50 times. Save the best one.
models = []
MAE = []
RMSE = []
for i in range(10):
    model = Sequential([
    Dense(units=10, input_shape=(3,), activation='softsign'),
    Dense(units=10, activation='softsign'),
    Dense(units=10, activation='softsign'),
    Dense(units=out_dim, activation = 'linear'),
    ])
    
    # fit the model 
    model.compile(loss='mse', optimizer='Adam', metrics=['mae'])
    model.fit(x=X_train,y=y_train,epochs=1000,validation_data=valid_set,batch_size=5,verbose=2)
    models.append(model)
    
    # analyze the model performance 
    p = model.predict(np.array(X_test))
    predicted_y = scaler.inverse_transform(p.reshape(-1,1))
    actual_y = scaler.inverse_transform(y_test.reshape(-1,1))
    mae = mean_absolute_error(actual_y, predicted_y)
    rmse = mean_squared_error(actual_y, predicted_y)

    #mae =  np.sum(np.absolute((predicted_y - actual_y))) / len(predicted_y)
    #pe = np.mean((predicted_y - actual_y) / actual_y)
    RMSE.append(rmse)
    MAE.append(mae)
    #percent_error.append(pe)
    print("Done with", i)

#models = np.asarray(models) 
MAE = np.asarray(MAE) 
print(MAE)
RMSE = np.asarray(RMSE) 
print(RMSE)
