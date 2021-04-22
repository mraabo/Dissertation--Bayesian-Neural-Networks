import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing

# Preparing data
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
# Normalizing
# to do

# Creating neural network
model = tf.keras.Sequential([
    tf.keras.Input((13, ), name='feature'),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=tf.nn.relu)
])
model.summary()

# Compile, train, and evaluate.
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])
model.fit(X_train, y_train, epochs=100)
model.evaluate(X_test, y_test)
