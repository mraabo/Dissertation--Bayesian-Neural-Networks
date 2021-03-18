import random as rn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# -------------------------------- Creating sin-data -------------------------------


def true_fun(x):
    return np.cos(1.5 * np.pi * x)


np.random.seed(42)
n_x = 6
x_train = np.sort(np.random.rand(n_x))
y_train = true_fun(x_train) + np.random.randn(n_x) * 0.1


# --------------------- Build and compile neural net -------------------------------

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=([1, ]), activation='tanh'),
    tf.keras.layers.Dense(1, activation='tanh')
])
model.summary()


# --------------------- Sample neural networks with acceptance-prop equal to likelihood ----------------------

n_NN = 200
weight_list = []
likelihood_list = []
sigma_k = 0.5  # sd of assumed gauss P(y \mid x), Neal's eq 1.8, Neal uses 0.1

# Sample neural networks and save their likelihood and weights
for i in range(n_NN):
    print(i)
    # sample weights with the right shape
    # Change later so output neuron-weight is sampled with sd = 1/sqrt(16)
    weights = [np.random.normal(0, 1, size=w.shape)
               for w in model.get_weights()]
    # set weight of net to those sampled above
    model.set_weights(weights)
    weight_list.append(weights)
    # Calculate gauss likelihood of y_train given weights and x_train, Neal's eq 1.11
    mean = model.predict(x_train)
    gauss = tfd.Normal(loc=mean, scale=sigma_k)
    likelihood = 1
    for j in range(n_x):
        likelihood *= gauss[j].prob(y_train[j])
    likelihood_list.append(likelihood)

# Normalize likelihood to max prob is 1
if max(likelihood_list) != 0:
    likelihood_list = likelihood_list/max(likelihood_list)

# Accept model with prob equal to normalized likelihood
accepted_weights = []
accepted_likelihood = []
for i in range(len(likelihood_list)):
    uniform_dist = tfd.Uniform(0, 1)
    if likelihood_list[i] >= uniform_dist.sample():
        accepted_weights.append(weight_list[i])
        accepted_likelihood.append(likelihood_list[i])


# --------------------- Use sampled networks for predicting y's ----------------------
x_pred = tf.linspace(0.0, 1, 200)
y_pred = []
for i in range(len(accepted_weights)):
    model.set_weights(accepted_weights[i])
    y_pred.append(model.predict(x_pred))

# --------------------- Plot of BNN results ----------------------
plt.scatter(x_train, y_train,
            edgecolor='b', s=20, label="Samples")
for i in range(len(y_pred)):
    plt.plot(x_pred, y_pred[i], color='k', label='Predictions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# Still need to plot average y_pred
