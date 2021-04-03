from HMC_NN_class import HMC_neural_network
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# --------------------------------- Parameters ------------------------------------
num_features = 2
num_examples = 50
num_hidden_neurons = 16
weight_prior_var = 1
weight_prior_mean = 0
weight_likelihood_stddev = 1.5

# --------------------------------- Creating toy data ------------------------------------


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

# --------------------------------- Instantiating class ------------------------------------

My_HMC_NN = HMC_neural_network(x_train, y_train, num_hidden_neurons,
                               weight_prior_mean, weight_prior_var, weight_likelihood_stddev, task="regression")

initial_state = np.ones(num_hidden_neurons * (num_features + 2)+1)

# --------------------------------- Test of methods ------------------------------------
hidden_w_num = num_hidden_neurons*(num_features + 1)
# Distributing weights to hidden and output layers with correct shape
hidden_w = tf.reshape(
    initial_state[0:hidden_w_num], [num_hidden_neurons, num_features + 1])
output_w = initial_state[hidden_w_num:]
print("network_func test:", My_HMC_NN.network_func(hidden_w, output_w))

print("joint_log_prob test:", My_HMC_NN.joint_log_prob(initial_state))

print("unnormalized_posterior test:",
      My_HMC_NN.unnormalized_posterior(initial_state))

My_HMC_NN.create_HMC_kernel()
samples, kernel_results = My_HMC_NN.run_chain(initial_state)
print("Acceptance rate:", kernel_results.is_accepted.numpy().mean())

y_pred = My_HMC_NN.predict(samples)
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
