import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import *


def NN_builder(layer_vector, optimizer, learning_rate, activation):
    """
    Builds a neural network

    Parameters
    ----------
    layer_vector : list
        A list describing the number of layers and the number of nodes per layer.
        e.g. for a model with 3 nodes in the input layer, 2 hidden layers with 10 nodes each, 
        and a single output node (3-10-10-1 system): layer_vector = [3, 10, 10, 1] 
    optimizer : str
        Which Keras optimizer to use
    learning_rate : str
        The learning rate for the optimizer
    activation : str
        The activation function
    """
    keras.backend.clear_session()
    nlayers = len(layer_vector)
    model = Sequential()

    for i in range(nlayers):
        if i == 0:
            model.add(Dense(units=layer_vector[1], input_shape=(layer_vector[0],), activation=activation))
        if i == (nlayers - 1):
            model.add(Dense(units=layer_vector[-1], activation=activation))
        else:
            model.add(Dense(units=layer_vector[i+1], activation=activation))
    model.compile(loss='mse', optimizer=optimizer(lr=learning_rate),metrics=['mse'])
    return model
