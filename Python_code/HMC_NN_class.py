import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class HMC_NN_regression:
    """ A class for a regression Bayesian neural network using Hamiltonian Monte-Carlo sampling """

    def __init__(self, x, y, num_hidden_neurons, weight_prior_mean, weight_prior_var, weight_likelihood_stddev):
        self.x = x
        self.num_features = x.shape[0]
        self.num_examples = x.shape[1]
        self.y = y
        self.num_hidden_neurons = num_hidden_neurons
        self.hidden_w_num = self.num_hidden_neurons*(self.num_features + 1)
        self.weight_prior_mean = weight_prior_mean
        self.weight_prior_var = weight_prior_var
        self.weight_likelihood_stddev = weight_likelihood_stddev

    def network_func(self, hidden_weights, output_weights):
        hidden_result = []
        # Pad x with 1's to add bias via matmul
        x_padded = tf.pad(self.x, [[1, 0], [0, 0]], constant_values=1)
        # Calculate x * w for each hidden neuron
        for i in range(hidden_weights.shape[0]):
            linop = tf.linalg.LinearOperatorFullMatrix(
                hidden_weights[i][..., np.newaxis])
            hidden_result.append(tf.keras.activations.tanh(
                linop.matmul(x_padded, adjoint=True)))

        # Single output neuron that uses linear activation function
        prod = []
        for i in range(self.num_hidden_neurons):
            prod.append(hidden_result[i]*output_weights[i])
        output_result = tf.math.add_n(prod) + output_weights[-1]
        return output_result[..., 0, :]

    def joint_log_prob(self, w):
        # Distributing weights to hidden and output layers with correct shape
        hidden_w = tf.reshape(
            w[0:self.hidden_w_num], [self.num_hidden_neurons, self.num_features + 1])
        output_w = w[self.hidden_w_num:]

        rv_w = tfd.MultivariateNormalDiag(
            loc=np.zeros(w.shape) + self.weight_prior_mean,
            scale_diag=np.ones(w.shape)*self.weight_prior_var)

        rv_y = tfd.Normal(self.network_func(
            hidden_w, output_w), self.weight_likelihood_stddev)
        y_logprob = rv_y.log_prob(self.y)
        return (rv_w.log_prob(w) + tf.reduce_sum(y_logprob, axis=-1))

    def unnormalized_posterior(self, w):
        return self.joint_log_prob(w)

    def create_HMC_kernel(self):
        # Create an HMC TransitionKernel
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.unnormalized_posterior,
            step_size=np.float64(.1),
            num_leapfrog_steps=0.5)

    # We wrap sample_chain in tf.function, telling TF to precompile a reusable
    # computation graph, which will dramatically improve performance.

    @ tf.function
    def run_chain(self, initial_state, num_results=1000, num_burnin_steps=500):
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=self.hmc_kernel,
            trace_fn=lambda current_state, kernel_results: kernel_results,
            seed=42)

    def predict(self, samples):
        sampled_hidden_w = samples[:, 0:self.hidden_w_num]
        sampled_output_w = samples[:, self.hidden_w_num:]

        # Get y-predictions using the sampled weights
        y_pred = np.zeros((sampled_hidden_w.shape[0], self.num_examples))
        for i in range(sampled_hidden_w.shape[0]):
            reshaped_sampled_hidden_w = tf.reshape(
                sampled_hidden_w[i], [self.num_hidden_neurons, self.num_features + 1])
            y_pred[i] = self.network_func(
                reshaped_sampled_hidden_w, sampled_output_w[i]).numpy()
        return y_pred


class HMC_NN_classification:
    """ A class for a classification Bayesian neural network using Hamiltonian Monte-Carlo sampling """

    def __init__(self, x, y, num_hidden_neurons, weight_prior_mean, weight_prior_var):
        self.x = x
        self.num_features = x.shape[0]
        self.num_examples = x.shape[1]
        self.y = y
        self.num_hidden_neurons = num_hidden_neurons
        self.hidden_w_num = self.num_hidden_neurons*(self.num_features + 1)
        self.weight_prior_mean = weight_prior_mean
        self.weight_prior_var = weight_prior_var
        self.unique_classes = np.unique(y)

    def network_func(self, hidden_weights, output_weights):
        hidden_result = []
        # Pad x with 1's to add bias via matmul
        x_padded = tf.pad(self.x, [[1, 0], [0, 0]], constant_values=1)
        # Calculate x * w for each hidden neuron
        for i in range(hidden_weights.shape[0]):
            linop = tf.linalg.LinearOperatorFullMatrix(
                hidden_weights[i][..., np.newaxis])
            hidden_result.append(tf.keras.activations.tanh(
                linop.matmul(x_padded, adjoint=True)))
        # Convert list to tensor
        hidden_result = tf.reshape(
            tf.convert_to_tensor(hidden_result), [self.num_hidden_neurons, self.num_examples])

        # Output neurons that uses softmax for classification
        # Pre allocate array of real-numbers
        output_prob = tf.TensorArray(
            tf.float64, size=0, dynamic_size=True)

        # # Old way to Calculate z_hidden * w_output
        # for j in range(len(self.unique_classes)):
        #     prod = []
        #     for i in range(self.num_hidden_neurons):
        #         prod.append(hidden_result[i]*output_weights[i])
        #     single_output_res = tf.math.add_n(prod) + output_weights[-1]
        #     output_reals = output_reals.write(
        #         j, single_output_res[..., 0, :])
        # output_reals = output_reals.stack()
        # Calculate softmax on real-numbers to get a probability for each output neuron
        # output_prob = tf.keras.activations.softmax(tf.transpose(
        #     tf.convert_to_tensor(output_reals)))

        # Pad hidden_result with 1's to add bias via matmul
        z_hidden_padded = tf.pad(
            hidden_result, [[1, 0], [0, 0]], constant_values=1)
        # Calculate z_hidden_padded * w_output for each hidden neuron
        for i in range(output_weights.shape[0]):
            linop = tf.linalg.LinearOperatorFullMatrix(
                output_weights[i][..., np.newaxis])
            output_prop = output_prob.write(i, tf.reshape(tf.keras.activations.softmax(
                linop.matmul(z_hidden_padded, adjoint=True)), [-1]))
        output_prob = output_prob.stack()
        return output_prob

    def joint_log_prob(self, w):
        # Distributing weights to hidden and output layers with correct shape
        hidden_w = tf.reshape(
            w[0:self.hidden_w_num], [self.num_hidden_neurons, self.num_features + 1])
        output_w = tf.reshape(w[self.hidden_w_num:], [
            len(self.unique_classes), self.num_hidden_neurons + 1])

        rv_w = tfd.MultivariateNormalDiag(
            loc=np.zeros(w.shape) + self.weight_prior_mean,
            scale_diag=np.ones(w.shape)*self.weight_prior_var)

        y_logprob = tf.TensorArray(
            tf.float64, size=0, dynamic_size=True)
        for i in range(len(self.y)):
            y_logprob = y_logprob.write(i, tf.math.log(self.network_func(
                hidden_w, output_w)[self.y[i] - 1, i]))
        y_logprob = y_logprob.stack()
        return (rv_w.log_prob(w) + tf.reduce_sum(y_logprob, axis=-1))

    def unnormalized_posterior(self, w):
        return self.joint_log_prob(w)

    def create_HMC_kernel(self):
        # Create an HMC TransitionKernel
        self.hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=self.unnormalized_posterior,
            step_size=np.float64(.1),
            num_leapfrog_steps=0.5)

    # We wrap sample_chain in tf.function, telling TF to precompile a reusable
    # computation graph, which will dramatically improve performance.

    @ tf.function
    def run_chain(self, initial_state, num_results=1000, num_burnin_steps=500):
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=self.hmc_kernel,
            trace_fn=lambda current_state, kernel_results: kernel_results,
            seed=42)

    def predict(self, samples):
        sampled_hidden_w = samples[:, 0:self.hidden_w_num]
        sampled_output_w = samples[:, self.hidden_w_num:]

        # Get y-predictions using the sampled weights
        y_pred = np.zeros((sampled_hidden_w.shape[0], self.num_examples))
        for i in range(sampled_hidden_w.shape[0]):
            reshaped_sampled_hidden_w = tf.reshape(
                sampled_hidden_w[i], [self.num_hidden_neurons, self.num_features + 1])
            y_pred[i] = self.network_func(
                reshaped_sampled_hidden_w, sampled_output_w[i]).numpy()
        # add ifelse for self.task, if classification add softmax and select most probably category
        return y_pred
