from keras.regularizers import l2
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
start_time = time.time()

# ----------------------------- Prepare data ---------------------------
credit_data = pd.read_csv(
    "Python_code/data/UCI_Credit_Card.csv", encoding="utf-8", index_col=0)
credit_data.head()

# Data to numpy
data = np.array(credit_data)

# Extract labels
data_X = data[:, 0:23]
data_y = data[:, 23]

data_X=data_X[0:100,:]
data_y=data_y[0:100]

X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.30, random_state=3030)

# ----------------------------- Neural Network ---------------------------
reg_const = 0.1
n_hidden = 10

model = tf.keras.Sequential([
    tf.keras.Input((23, ), name='feature'),
    tf.keras.layers.Dense(n_hidden, activation=tf.nn.relu, kernel_regularizer=l2(
        reg_const), bias_regularizer=l2(reg_const)),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
model.summary()

# Compile, train, and evaluate.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=200,
                    validation_split=0.3)

model.evaluate(X_test, y_test)

print("--- %s seconds ---" % (time.time() - start_time))

# ----------------------------- Heatmap ---------------------------

# Predict class 1 for prob > 0.25 and class 0 otherwise
y_pred_test = model.predict(X_test) > 0.1
conf_mat = confusion_matrix(y_test, y_pred_test, normalize='all')
sns.heatmap(conf_mat, cmap=plt.cm.Blues, annot=True)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()

# ----------------------------- Overfitting? ---------------------------

train_acc = model.evaluate(X_train, y_train, verbose=0)[-1]
test_acc = model.evaluate(X_test, y_test, verbose=0)[-1]
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.legend()
plt.grid()
plt.show()
