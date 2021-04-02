import random as rn
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


class HMC_neural_network:
    """ A class for a Bayesian neural network using Hamiltonian Monte-Carlo sampling """

    def __init__(self, x, y, num_hidden_neurons, weight_prior_mean, weight_prior_var, weight_likelihood_stddev, task='regression'):
        self.x = x
        self.num_features = x.shape[0]
        self.num_examples = x.shape[1]
        self.y = y
        self.num_hidden_neurons = num_hidden_neurons
        self.hidden_w_num = self.num_hidden_neurons*(self.num_features + 1)
        self.weight_prior_mean = weight_prior_mean
        self.weight_prior_var = weight_prior_var
        self.weight_likelihood_stddev = weight_likelihood_stddev
        self.task = task

    def network_func(self, hidden_weights, output_weights):
        hidden_result = []
        # Pad x with 1's to add bias via matmul
        x_padded = tf.pad(self.x, [[1, 0], [0, 0]], constant_values=1)
        for i in range(hidden_weights.shape[0]):
            linop = tf.linalg.LinearOperatorFullMatrix(
                hidden_weights[i][..., np.newaxis])
            hidden_result.append(tf.keras.activations.tanh(
                linop.matmul(x_padded, adjoint=True)))

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
        # add ifelse for self.task, if classification make softmax likelihood
        rv_y = tfd.Normal(self.network_func(
            hidden_w, output_w), self.weight_likelihood_stddev)
        return (rv_w.log_prob(w) + tf.reduce_sum(rv_y.log_prob(self.y), axis=-1))

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
    def run_chain(self, initial_state, num_results=5000, num_burnin_steps=500):
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
