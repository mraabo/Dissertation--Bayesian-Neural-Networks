import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


num_features = 2
num_examples = 50
num_hidden_neurons = 16
hidden_w_num = num_hidden_neurons*(num_features + 1)
weight_prior_var = 1
weight_prior_mean = 0
weight_likelihood_stddev = 1.5
# -------------------------------- Creating data -------------------------------


# def true_fun(x):
#     return np.sin(3 * x)  # np.sin(1.5 * np.pi * x)


# np.random.seed(42)
# n_x = 6
# x_train = np.sort(np.random.rand(n_x))
# y_train = true_fun(x_train)  # + np.random.randn(n_x) * 0.1

# Generate some data
def f(x, w):
    # Pad x with 1's so we can add bias via matmul
    x = tf.pad(x, [[1, 0], [0, 0]], constant_values=1)
    linop = tf.linalg.LinearOperatorFullMatrix(w[..., np.newaxis])
    result = linop.matmul(x, adjoint=True)
    return result[..., 0, :]


noise_scale = .5
true_w = np.array([-1., 2., 3.])

np.random.seed(42)
x_train = np.random.uniform(-1., 1., [num_features, num_examples])
y_train = f(x_train, true_w) + np.random.normal(0.,
                                                noise_scale, size=num_examples)

# --------------------- Build and compile neural net -------------------------------

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, input_shape=([1, ]), activation='tanh'),
    tf.keras.layers.Dense(1, activation='tanh')
])
model.summary()


# --------------------- Sample weights using Hamiltonian Monte Carlo ----------------------
def network_func(hidden_weights, output_weights, x):

    for layer in model.layers:
        if layer.name == "dense":
            layer.set_weights(hidden_weights)
        if layer.name == "dense_1":
            layer.set_weights(output_weights)
    return model.predict(x)

# def network_func(hidden_weights, output_weights, x):
#     hidden_result = []
#     # Pad x with 1's to add bias via matmul
#     x = tf.pad(x, [[1, 0], [0, 0]], constant_values=1)
#     for i in range(hidden_weights.shape[0]):
#         linop = tf.linalg.LinearOperatorFullMatrix(
#             hidden_weights[i][..., np.newaxis])
#         hidden_result.append(tf.keras.activations.tanh(
#             linop.matmul(x, adjoint=True)))

#     prod = []
#     for i in range(num_hidden_neurons):
#         prod.append(hidden_result[i]*output_weights[i])
#     output_result = tf.math.add_n(prod) + output_weights[-1]
#     return output_result[..., 0, :]


def joint_log_prob(w, x, y):
    # Distributing weights to hidden and output layers with correct shape
    hidden_w = tf.reshape(
        w[0:hidden_w_num], [num_hidden_neurons, num_features + 1])
    output_w = w[hidden_w_num:]

    rv_w = tfd.MultivariateNormalDiag(
        loc=np.zeros(w.shape) + weight_prior_mean,
        scale_diag=np.ones(w.shape)*weight_prior_var)

    rv_y = tfd.Normal(network_func(hidden_w, output_w,
                                   x_train), weight_likelihood_stddev)
    return (rv_w.log_prob(w) + tf.reduce_sum(rv_y.log_prob(y), axis=-1))


def unnormalized_posterior(w):
    return joint_log_prob(w, x_train, y_train)


# Create an HMC TransitionKernel
hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=unnormalized_posterior,
    step_size=np.float64(.1),
    num_leapfrog_steps=0.5)

# We wrap sample_chain in tf.function, telling TF to precompile a reusable
# computation graph, which will dramatically improve performance.


@ tf.function
def run_chain(initial_state, num_results=5000, num_burnin_steps=500):
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        num_burnin_steps=num_burnin_steps,
        current_state=initial_state,
        kernel=hmc_kernel,
        trace_fn=lambda current_state, kernel_results: kernel_results,
        seed=42)


initial_state = np.ones(num_hidden_neurons * (num_features + 2)+1)
samples, kernel_results = run_chain(initial_state)
print("Acceptance rate:", kernel_results.is_accepted.numpy().mean())


sampled_hidden_w = samples[:, 0:hidden_w_num]
sampled_output_w = samples[:, hidden_w_num:]

# Get y-predictions using the sampled weights
y_pred = np.zeros((sampled_hidden_w.shape[0], num_examples))
for i in range(sampled_hidden_w.shape[0]):
    reshaped_sampled_hidden_w = tf.reshape(
        sampled_hidden_w[i], [num_hidden_neurons, num_features + 1])
    y_pred[i] = network_func(
        reshaped_sampled_hidden_w, sampled_output_w[i], x_train).numpy()

y_pred_mean = y_pred.mean(axis=0)


print("MSE:", np.sum((y_pred_mean - y_train)**2))


# Visualize the data set
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x_train[0], x_train[1], y_train,
             c=y_train, cmap='copper', marker='o')
ax.scatter3D(x_train[0], x_train[1], y_pred_mean,
             c=y_pred_mean, cmap='copper', marker='x')
plt.show()
