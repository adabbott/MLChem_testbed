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
    X_train, X_fulltest, y_train, y_fulltest = train_test_split(scaled_X, scaled_y, test_size = 0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_fulltest, y_fulltest, test_size = 0.5, random_state=42)
    return X_train, X_valid, X_test, y_train, y_valid, y_test
    

def create_model(X_train, X_valid, X_test, y_train, y_valid, y_test):
    in_dim = tuple([X_train.shape[1]])
    out_dim = y_train.shape[1]
    valid_set = tuple([X_valid, y_valid])

    model = Sequential()
    model.add(Dense(  {{choice([25,50,100])}}, input_shape=in_dim))
    model.add(Activation(   {{choice(['sigmoid', 'softmax', 'linear'])}}  ))
    model.add(Dense(  {{choice([25,50,100])}}  ))
    model.add(Activation(   {{choice(['sigmoid','softmax', 'linear'])}}  ))
    
    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(  {{choice([25,50,100])}}  ))
        model.add(Activation(   {{choice(['sigmoid','softmax', 'linear'])}}  ))
        if {{choice(['four', 'five'])}} == 'five':
            model.add(Dense(  {{choice([25,50,100])}}  ))
            model.add(Activation(   {{choice(['sigmoid','softmax', 'linear'])}}  ))
            if {{choice(['five', 'six'])}} == 'six':
                model.add(Dense(  {{choice([25,50,100])}}  ))
                model.add(Activation(   {{choice(['sigmoid','softmax', 'linear'])}}  ))
    model.add(Dense(out_dim))
    # hyperas bugs out if you put a choice in last layer :(
    model.add(Activation('linear'))
    
    model.compile(loss='mse', optimizer={{choice(['Adam', 'Adagrad', 'Nadam', 'Adadelta'])}}, metrics=['mae'])
    model.fit(x=X_train,y=y_train,epochs=500,validation_data=valid_set,batch_size=10,verbose=2)

    loss, mae = model.evaluate(X_test, y_test)
    print("Test RMSE: ", loss)
    print("Test MAE: ", mae)
    return {'loss': loss, 'status': STATUS_OK, 'model':model}

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          trials=Trials())
    scaler, scaled_X, scaled_y = scaling()
    X_train, X_valid, X_test, y_train, y_valid, y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, y_test))
 

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
    
    # RESULTS WRITING
    with open("RESULTS.txt", "w+") as f:
        f.write("Best performing model hyperparameters:\n")
        f.write(str(sorted(best_run.items())))
        f.write("\n\nModel Results (scaled data):\n")
        f.write("Test dataset RMSE: {}\n".format(rmse))
        f.write("Test dataset MAE: {}\n".format(mae))
        # write model information here
        f.write("\n\nModel Results (units- Hartree):\n")
        f.write("Test dataset RMSE: {}\n".format(rmse))
        f.write("Test dataset MAE: {}\n".format(mae))
        f.write("Full dataset RMSE: {}\n".format(rmse_f))
        f.write("Full dataset MAE: {}\n".format(mae_f))

    # MODEL SAVING
    best_model.save("best_model.h5")
    print("Saved best model from hyperparameter search to disk")
    
    

