import random as rn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# -------------------------------- Creating sin-data -------------------------------


def true_fun(x):
    return np.sin(3 * x)  # np.sin(1.5 * np.pi * x)


np.random.seed(42)
n_x = 6
x_train = np.sort(np.random.rand(n_x))
y_train = true_fun(x_train)  # + np.random.randn(n_x) * 0.1


# --------------------- Build and compile neural net -------------------------------

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=([1, ]), activation='tanh'),
    tf.keras.layers.Dense(1, activation='tanh')
])
model.summary()


# --------------------- Sample weights through neural networks with acceptance-prop equal to likelihood ----------------------

n_NN = 10**5
weight_list = []
likelihood_list = []
sigma_k = 0.1  # sd of assumed gauss P(y \mid x), Neal's eq 1.8, Neal uses 0.1
tf.random.set_seed(42)
# Sample neural networks and save their likelihood and weights
for i in range(n_NN):
    print(i)
    # sample weights and set them to hidden layer "dense" and output layer "dense_1"
    for layer in model.layers:
        if layer.name == "dense":
            layer.set_weights([np.random.normal(0, 8, size=w.shape)
                               for w in layer.get_weights()])
        if layer.name == "dense_1":
            layer.set_weights([np.random.normal(
                0, 1/np.sqrt(16), size=w.shape) for w in layer.get_weights()])
    # save weights in list
    weight_list.append(model.get_weights())
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


# --------------------- Use sampled weights for predicting y's ----------------------
x_pred = tf.linspace(0.0, 1, 200)
y_pred = []
for i in range(len(accepted_weights)):
    model.set_weights(accepted_weights[i])
    y_pred.append(model.predict(x_pred))

# mean y_pred
mean_y_pred = np.array(y_pred).mean(axis=0)

# --------------------- Plot of BNN results ----------------------
plt.scatter(x_train, y_train,
            edgecolor='b', s=20, label="Samples")
for i in range(len(y_pred)):
    plt.plot(x_pred, y_pred[i], color='k', linestyle='dashed')
plt.plot(x_pred, mean_y_pred, label="Average prediction")
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.savefig('BNN_radfords_fig1.2.pdf')
plt.show()
