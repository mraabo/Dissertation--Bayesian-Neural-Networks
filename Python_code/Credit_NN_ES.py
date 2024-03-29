import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns

tf.random.set_seed(40)
# ----------------------------- Prepare data ---------------------------
credit_data = pd.read_csv(
    "Python_code/data/UCI_Credit_Card.csv", encoding="utf-8", index_col=0)
credit_data.head()

# Data to numpy
data = np.array(credit_data)

# Extract labels
data_X = data[:, 0:23]
data_y = data[:, 23]


# # ----------------------------- Subsamling credit data ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.30, random_state=3030)

N = 300
N_test = 100
X_train = X_train[0:N, :]
y_train = y_train[0:N]
X_test = X_test[0:N_test, :]
y_test = y_test[0:N_test]


# ----------------------------- Neural Network ---------------------------

model = tf.keras.Sequential([
    tf.keras.Input((23, ), name='feature'),
    tf.keras.layers.Dense(10, activation=tf.nn.tanh),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.summary()

# Early stopping
es = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', mode='min', patience=0, min_delta=0)

start_time = time.time()

# Compile, train, and evaluate.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_crossentropy'])
history = model.fit(X_train, y_train,  validation_split=0.3,
                    epochs=1000, callbacks=[es])
print("The algorithm ran", len(history.history['loss']), "epochs")

print("--- %s seconds ---" % (time.time() - start_time))


# ----------------------------- Overfitting? ---------------------------
train_acc = model.evaluate(X_train, y_train, verbose=0)[-1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[-1]
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

# taking mean of summed cross-entropy loss
train_loss = np.array(history.history['loss'])
val_loss = np.array(history.history['val_loss'])

plt.plot(train_loss, label='train')
plt.plot(val_loss, label='validation')
plt.legend()
plt.grid()
plt.show()
