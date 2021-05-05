import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
import time
start_time = time.time()

tf.random.set_seed(42)

# ----------------------------- Prepare data ---------------------------
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()

# pad Xs with 1's to add bias
ones_train=np.ones(X_train.shape[0])
ones_test=np.ones(X_test.shape[0])
X_train=np.insert(X_train,0,ones,axis=1)
X_test=np.insert(X_test,0,ones,axis=1)

# ----------------------------- Neural Network ---------------------------
model = tf.keras.Sequential([
    tf.keras.Input((13, ), name='feature'),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(1)
])
model.summary()

# Early stopping
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=20, min_delta=1)

# Compile, train, and evaluate.
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mse'])
history = model.fit(X_train, y_train, epochs=500,
                    validation_split=0.3, callbacks=[es], shuffle=False)

print("The algorithm ran", len(history.history['loss']), "epochs")

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
