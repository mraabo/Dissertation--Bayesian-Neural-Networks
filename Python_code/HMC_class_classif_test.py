from HMC_NN_class import HMC_NN_classification
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# --------------------------------- Parameters ------------------------------------
num_features = 2
num_examples = 4
num_hidden_neurons = 16
weight_prior_var = 1
weight_prior_mean = 0
weight_likelihood_stddev = 1.5

# --------------------------------- Creating toy data ------------------------------------
x = np.array([[1., 2.], [21., 22.], [51., 52.], [53, 54]]).T
y = np.array([1, 2, 3, 3])

# --------------------------------- Instantiating class ------------------------------------
My_HMC_NN = HMC_NN_classification(x, y, num_hidden_neurons,
                                  weight_prior_mean, weight_prior_var)

initial_state = np.ones(num_hidden_neurons * (num_features + 1) +
                        len(np.unique(y)) * (num_hidden_neurons + 1))

# --------------------------------- Test of methods ------------------------------------
# hidden_w_num = num_hidden_neurons*(num_features + 1)
# hidden_w = tf.reshape(
#     initial_state[0:hidden_w_num], [num_hidden_neurons, num_features + 1])
# output_w = tf.reshape(initial_state[hidden_w_num:], [
#                       len(np.unique(y)), num_hidden_neurons + 1])
# print("network_func test:", My_HMC_NN.network_func(hidden_w, output_w))

# print("joint_log_prob test:", My_HMC_NN.joint_log_prob(initial_state))

# print("unnormalized_posterior test:",
#       My_HMC_NN.unnormalized_posterior(initial_state))

My_HMC_NN.create_HMC_kernel()
samples, kernel_results = My_HMC_NN.run_chain(initial_state)
print("Acceptance rate:", kernel_results.is_accepted.numpy().mean())
