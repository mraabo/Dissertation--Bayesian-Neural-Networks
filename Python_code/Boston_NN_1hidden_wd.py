import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import time
from keras.regularizers import l2

tf.random.set_seed(40)
# ----------------------------- Prepare data ---------------------------
(X_train, y_train), (X_test, y_test) = boston_housing.load_data(seed=3030)


# ----------------------------- Neural Network ---------------------------
reg_const = 0.3
n_hidden = 10

model = tf.keras.Sequential([
    tf.keras.Input((13, ), name='feature'),
    tf.keras.layers.Dense(n_hidden, activation=tf.nn.relu, kernel_regularizer=l2(
        reg_const), bias_regularizer=l2(reg_const)),
    tf.keras.layers.Dense(1, kernel_regularizer=l2(
        reg_const), bias_regularizer=l2(reg_const))
])
model.summary()

start_time = time.time()
# Compile, train, and evaluate.
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])
history = model.fit(X_train, y_train, epochs=300, validation_split=0.3)
model.evaluate(X_test, y_test)

print("--- %s seconds ---" % (time.time() - start_time))

# ----------------------------- Overfitting? ---------------------------
train_acc = model.evaluate(X_train, y_train, verbose=0)[-1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[-1]
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.grid()
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.ylim(0, 200)
plt.savefig('figure_Boston_NN_1hidden_wd_loss.pdf')
plt.show()
