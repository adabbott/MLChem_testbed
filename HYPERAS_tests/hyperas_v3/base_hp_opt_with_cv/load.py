from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pandas as pd
best_model = load_model("best_model.h5")
print("Loaded model from disk")

def scaling():
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
    return scaler, scaled_X, scaled_y

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
    X_train, X_fulltest, y_train, y_fulltest = train_test_split(scaled_X, scaled_y, test_size = 0.5, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_fulltest, y_fulltest, test_size = 0.5, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

X_train, X_valid, X_test, y_train, y_valid, y_test = data()
scaler, scaled_X, scaled_y = scaling()
valid_set = tuple([X_valid, y_valid])
history = best_model.fit(x=X_train,y=y_train,epochs=1000,validation_data=valid_set,batch_size=5,verbose=2)


#RESULTS PRINTING
predicted_scaled_y = best_model.predict(X_test)
predicted_y = scaler.inverse_transform(predicted_scaled_y.reshape(-1,1))
actual_y = scaler.inverse_transform(y_test.reshape(-1,1))
mae = mean_absolute_error(actual_y, predicted_y)
rmse = mean_squared_error(actual_y, predicted_y)
print("Test dataset RMSE:", rmse)
print("Test dataset MAE:", mae)

predicted_scaled_y = best_model.predict(scaled_X)
predicted_y = scaler.inverse_transform(predicted_scaled_y.reshape(-1,1))
actual_y = scaler.inverse_transform(scaled_y.reshape(-1,1))
mae_f = mean_absolute_error(actual_y, predicted_y)
rmse_f = mean_squared_error(actual_y, predicted_y)
print("Full dataset RMSE:", rmse_f)
print("Full dataset MAE:", mae_f)

# PLOT 
import matplotlib.pyplot as plt
t = [1 - i for i in history.history['mean_absolute_error']]
v = [1 - i for i in history.history['val_mean_absolute_error']]
plt.plot(t)
plt.plot(v)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.show()
