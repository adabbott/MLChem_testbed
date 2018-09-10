from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.models import load_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy
numpy.set_printoptions(threshold=10000, linewidth=150)


# to be used later
#opt = Adam(lr={{uniform(0.0001,0.1)}},
#                                 beta_1=0.9,
#                                 beta_2=0.999,
#                                 epsilon=None,
#                                 decay=0.0,
#                                 amsgrad={{choice([True, False])}})



def data():
    data = pd.read_csv("PES.dat") 
    # drop duplicates, toggle on or off
    data = data.drop_duplicates(subset = 'E')
    data = data.values
    X = data[:,0:-1]
    y = data[:,-1].reshape(-1,1)
    return X,y

# MinMax scaling, 3 Fold CV, up to 5 layers, standard Adam optimizer default settings, 500 epochs. 
# This HyperOpt is for neural architecture search, while optimizer parameters (lr, decay, amsgrad) will be searched across later.
def create_model(X,y):
    scaler = MinMaxScaler(feature_range=(-1,1))
    sX = scaler.fit_transform(X)
    sy = scaler.fit_transform(y)

    in_dim = tuple([sX.shape[1]])
    out_dim = sy.shape[1]
    kfold = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_losses = []
    cv_maes = []
    for train, test in kfold.split(sX, sy):
        model = Sequential()
        model.add(Dense(50, input_shape=in_dim))
        model.add(Activation('linear'))
        model.add(Dense(100))
        model.add(Activation('tanh'))
        model.add(Dense(100))
        model.add(Activation('sigmoid'))
        model.add(Dense(out_dim))
        model.add(Activation('linear'))

        opt = Adam(lr={{uniform(0.01,0.1)}},
                                         beta_1=0.9,
                                         beta_2=0.999,
                                         epsilon=None,
                                         decay={{choice([0.00001, 0.000001, 0.0000001, 0.0])}},
                                         amsgrad=True)

        model.compile(loss='mse', optimizer=opt, metrics=['mae'])
        model.fit(x=sX,y=sy,epochs=2000,batch_size=10,verbose=0)

        # since comparing scaling methods, track model performance wrt actual energies
        predicted_scaled_y = model.predict(sX[test])
        predicted_y = scaler.inverse_transform(predicted_scaled_y.reshape(-1,1))
        actual_y = scaler.inverse_transform(sy[test])
        mae = mean_absolute_error(actual_y, predicted_y)
        loss = mean_squared_error(actual_y, predicted_y)

        #loss, mae = model.evaluate(sX[test], sy[test],verbose=0)
        cv_losses.append(loss)
        cv_maes.append(mae)
    print("Test MSEs: ", cv_losses)
    print("Test MAEs: ", cv_maes)
    loss = numpy.mean(cv_losses)
    # this model is just the last model in CV, not necessarily the best
    return {'loss': loss, 'status': STATUS_OK, 'model':model}


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=20,
                                          trials=Trials(),
                                          eval_space=True)
    with open("RESULTS.txt", "w+") as f:
        f.write("Best performing model hyperparameters:\n")
        f.write(str(sorted(best_run.items())))
        best_model.save("best_model.h5")
        print("Saved best model from hyperparameter search to disk")
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
    
#    scaler, scaled_X, scaled_y = scaling()
#    predicted_scaled_y = best_model.predict(scaled_X)
#    predicted_y = scaler.inverse_transform(predicted_scaled_y.reshape(-1,1))
#    actual_y = scaler.inverse_transform(scaled_y.reshape(-1,1))
#    mae_f = mean_absolute_error(actual_y, predicted_y)
#    rmse_f = mean_squared_error(actual_y, predicted_y)
#    print("Full dataset RMSE:", rmse_f)
#    print("Full dataset MAE:", mae_f)
#    
#    # RESULTS WRITING
#    with open("RESULTS.txt", "w+") as f:
#        f.write("Best performing model hyperparameters:\n")
#        f.write(str(sorted(best_run.items())))
#        #f.write("\n\nModel Results (scaled data):\n")
#        #f.write("Test dataset RMSE: {}\n".format(rmse))
#        #f.write("Test dataset MAE: {}\n".format(mae))
#        # write model information here
#        f.write("\n\nModel Results (units- Hartree):\n")
#        f.write("Full dataset RMSE: {}\n".format(rmse_f))
#        f.write("Full dataset MAE: {}\n".format(mae_f))
#
#    # MODEL SAVING
#    # this model is just the last model in CV, not necessarily the best. It is the best w.r.t. hyperparameters
#    best_model.save("best_model.h5")
#    print("Saved best model from hyperparameter search to disk")
    
    

