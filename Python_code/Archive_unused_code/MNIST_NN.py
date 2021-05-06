import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# Preparing data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# Normalizing
x_train, x_test = x_train / 255.0, x_test / 255.0

# Creating neural network
model = tf.keras.Sequential([
    tf.keras.Input((28, 28), name='feature'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.summary()

# Compile, train, and evaluate.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=32, epochs=5)
model.evaluate(x_test, y_test)
