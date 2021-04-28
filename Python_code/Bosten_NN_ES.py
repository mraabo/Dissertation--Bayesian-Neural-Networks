import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import boston_housing
from sklearn.preprocessing import StandardScaler
import time
start_time = time.time()
sc = StandardScaler()
# ----------------------------- Prepare data ---------------------------
(X_train, y_train), (X_test, y_test) = boston_housing.load_data()
# Standardizing
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

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
                    validation_split=0.3, callbacks=[es])

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
