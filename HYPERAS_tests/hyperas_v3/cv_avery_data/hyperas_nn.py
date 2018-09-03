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
import numpy

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

def original_data():
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


def data():
    data = pd.read_csv("PES.dat") 
    # drop duplicates, toggle on or off
    data = data.drop_duplicates(subset = 'E')
    data = data.values
    X = data[:,0:-1]
    y = data[:,-1]
    # should you scale a smaller range to fit multiple activ functions?
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(X)
    y = scaler.fit_transform(y.reshape(-1,1))
    return X,y  
    

#def create_model(X_train, X_valid, X_test, y_train, y_valid, y_test):
def create_model(X,y):
    in_dim = tuple([X.shape[1]])
    out_dim = y.shape[1]

    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_losses = []
    cv_maes = []
    for train, test in kfold.split(X, y):
        model = Sequential()
        model.add(Dense(  {{choice([25,50,100])}}, input_shape=in_dim))
        model.add(Activation(   {{choice(['sigmoid', 'softmax', 'linear'])}}  ))
        model.add(Dense(  {{choice([25,50,100])}}  ))
        model.add(Activation(   {{choice(['sigmoid','softmax', 'linear'])}}  ))

        if {{choice(['three', 'four'])}} == 'four':
            model.add(Dense(  {{choice([25,50,100])}}  ))
            model.add(Activation(   {{choice(['sigmoid','softmax', 'linear'])}}  ))
        model.add(Dense(out_dim))
        model.add(Activation('linear'))
        
        model.compile(loss='mse', optimizer={{choice(['Adam'])}}, metrics=['mae'])
        model.fit(x=X,y=y,epochs=500,batch_size=15,verbose=0)
        loss, mae = model.evaluate(X[test], y[test],verbose=0)
        cv_losses.append(loss)
        cv_maes.append(mae)
    print("Test RMSEs: ", cv_losses)
    print("Test MAEs: ", cv_maes)
    loss = numpy.mean(cv_losses)
    # this model is just the last model in CV, not necessarily the best
    return {'loss': loss, 'status': STATUS_OK, 'model':model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=50,
                                          trials=Trials())
#    scaler, scaled_X, scaled_y = scaling()
#    X_train, X_valid, X_test, y_train, y_valid, y_test = data()
#    print("Evalutation of best performing model:")
#    print(best_model.evaluate(X_test, y_test))
# 
#
#    #RESULTS PRINTING
#    predicted_scaled_y = best_model.predict(X_test)
#    predicted_y = scaler.inverse_transform(predicted_scaled_y.reshape(-1,1))
#    actual_y = scaler.inverse_transform(y_test.reshape(-1,1))
#    mae = mean_absolute_error(actual_y, predicted_y)
#    rmse = mean_squared_error(actual_y, predicted_y)
#    print("Test dataset RMSE:", rmse)
#    print("Test dataset MAE:", mae)
    
    scaler, scaled_X, scaled_y = scaling()
    predicted_scaled_y = best_model.predict(scaled_X)
    predicted_y = scaler.inverse_transform(predicted_scaled_y.reshape(-1,1))
    actual_y = scaler.inverse_transform(scaled_y.reshape(-1,1))
    mae_f = mean_absolute_error(actual_y, predicted_y)
    rmse_f = mean_squared_error(actual_y, predicted_y)
    print("Full dataset RMSE:", rmse_f)
    print("Full dataset MAE:", mae_f)
    
    # RESULTS WRITING
    with open("RESULTS.txt", "w+") as f:
        f.write("Best performing model hyperparameters:\n")
        f.write(str(sorted(best_run.items())))
        #f.write("\n\nModel Results (scaled data):\n")
        #f.write("Test dataset RMSE: {}\n".format(rmse))
        #f.write("Test dataset MAE: {}\n".format(mae))
        # write model information here
        f.write("\n\nModel Results (units- Hartree):\n")
        f.write("Full dataset RMSE: {}\n".format(rmse_f))
        f.write("Full dataset MAE: {}\n".format(mae_f))

    # MODEL SAVING
    # this model is just the last model in CV, not necessarily the best. It is the best w.r.t. hyperparameters
    best_model.save("best_model.h5")
    print("Saved best model from hyperparameter search to disk")
    
    

