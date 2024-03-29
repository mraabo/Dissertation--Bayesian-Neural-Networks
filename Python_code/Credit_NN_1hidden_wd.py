from keras.regularizers import l2
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
start_time = time.time()
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
reg_const = 0.1
n_hidden = 10

model = tf.keras.Sequential([
    tf.keras.Input((23, ), name='feature'),
    tf.keras.layers.Dense(n_hidden, activation=tf.nn.tanh, kernel_regularizer=l2(
        reg_const), bias_regularizer=l2(reg_const)),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, kernel_regularizer=l2(
        reg_const), bias_regularizer=l2(reg_const))
])
model.summary()

# Compile, train, and evaluate.
val_ratio = 0.3
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['binary_crossentropy'])
history = model.fit(X_train, y_train, epochs=1000,
                    validation_split=val_ratio)

model.evaluate(X_test, y_test)

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
plt.ylim(0.4, 1)
#plt.savefig('Python_code/figure_Credit_NN_1hidden_wd_loss.pdf')
plt.show()
