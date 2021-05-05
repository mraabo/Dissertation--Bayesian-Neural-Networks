import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import time


# ----------------------------- Prepare data ---------------------------
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()


# ----------------------------- Neural Network ---------------------------
model = tf.keras.Sequential([
    tf.keras.Input((13, ), name='feature'),
    tf.keras.layers.Dense(15, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
])
model.summary()

start_time = time.time()
# Compile, train, and evaluate.
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])
history = model.fit(X_train, y_train, epochs=200, validation_split=0.3)
model.evaluate(X_test, y_test)

print("--- %s seconds ---" % (time.time() - start_time))

# ----------------------------- Overfitting? ---------------------------
train_acc = model.evaluate(X_train, y_train, verbose=0)[-1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[-1]
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.grid()
plt.show()
