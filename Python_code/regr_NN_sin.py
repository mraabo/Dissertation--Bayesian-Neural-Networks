import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rn

# -------------------------------- Creating sin-data -------------------------------


def true_fun(x):
    return np.cos(1.5 * np.pi * x)


np.random.seed(42)
n_samples = 50
x_train = np.sort(np.random.rand(n_samples))
y_train = true_fun(x_train) + np.random.randn(n_samples) * 0.1


# --------------------- Build and compile neural net -------------------------------

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=([1, ]), activation='tanh'),
    tf.keras.layers.Dense(1, activation='tanh')
])
model.summary()
print(model.predict(x_train))

model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(0.001))


# --------------------- training and results of neural net --------------------------

model.fit(x_train, y_train, epochs=500)

x_pred = tf.linspace(0.0, 1, n_samples)
y_pred = model.predict(x_pred)
print(model.layers[1].kernel)

plt.scatter(x_train, y_train,
            edgecolor='b', s=20, label="Samples")
plt.plot(x_pred, y_pred, color='k', label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("done")
