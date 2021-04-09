from autograd import grad
import autograd.numpy as np
import scipy.stats as st
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions
# ------ Credit to Colin Carroll


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

    def hamiltonian_monte_carlo(self, n_samples, initial_position, path_len=1, step_size=0.5):
        """Run Hamiltonian Monte Carlo sampling.

        Parameters
        ----------
        n_samples : int
            Number of samples to return
        negative_log_prob : callable
            The negative log probability to sample from
        initial_position : np.array
            A place to start sampling from.
        path_len : float
            How long each integration path is. Smaller is faster and more correlated.
        step_size : float
            How long each integration step is. Smaller is slower and more accurate.

        Returns
        -------
        np.array
            Array of length `n_samples`.
        """
        # autograd magic
        dVdq = grad(self.joint_log_prob)

        # collect all our samples in a list
        samples = [initial_position]

        # Keep a single object for momentum resampling
        momentum = st.norm(0, 1)

        # If initial_position is a 10d vector and n_samples is 100, we want
        # 100 x 10 momentum draws. We can do this in one call to momentum.rvs, and
        # iterate over rows
        size = (n_samples,) + initial_position.shape[:1]
        for p0 in momentum.rvs(size=size):
            # Integrate over our path to get a new position and momentum
            q_new, p_new = self.leapfrog(
                samples[-1],
                p0,
                dVdq,
                path_len=path_len,
                step_size=step_size,
            )

            # Check Metropolis acceptance criterion
            start_log_p = self.joint_log_prob(
                samples[-1]) - np.sum(momentum.logpdf(p0))
            new_log_p = self.joint_log_prob(
                q_new) - np.sum(momentum.logpdf(p_new))
            if np.log(np.random.rand()) < start_log_p - new_log_p:
                samples.append(q_new)
            else:
                samples.append(np.copy(samples[-1]))

        return np.array(samples[1:])

    def leapfrog(self, q, p, dVdq, path_len, step_size):
        """Leapfrog integrator for Hamiltonian Monte Carlo.

        Parameters
        ----------
        q : np.floatX
            Initial position
        p : np.floatX
            Initial momentum
        dVdq : callable
            Gradient of the velocity
        path_len : float
            How long to integrate for
        step_size : float
            How long each integration step should be

        Returns
        -------
        q, p : np.floatX, np.floatX
            New position and momentum
        """
        q, p = np.copy(q), np.copy(p)

        p -= step_size * dVdq(q) / 2  # half step
        for _ in range(int(path_len / step_size) - 1):
            q += step_size * p  # whole step
            p -= step_size * dVdq(q)  # whole step
        q += step_size * p  # whole step
        p -= step_size * dVdq(q) / 2  # half step

        # momentum flip at end
        return q, -p

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


############################################################################################################
#                                                                                                          #
#                                                                                                          #
#                                           Test with regression                                           #
#                                                                                                          #
#                                                                                                          #
############################################################################################################

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

My_HMC_NN = HMC_NN_regression(x_train, y_train, num_hidden_neurons,
                              weight_prior_mean, weight_prior_var, weight_likelihood_stddev)

initial_state = np.ones(num_hidden_neurons * (num_features + 2)+1)

# --------------------------------- Test of methods ------------------------------------
hidden_w_num = num_hidden_neurons*(num_features + 1)
# Distributing weights to hidden and output layers with correct shape
hidden_w = tf.reshape(
    initial_state[0:hidden_w_num], [num_hidden_neurons, num_features + 1])
output_w = initial_state[hidden_w_num:]
print("network_func test:", My_HMC_NN.network_func(hidden_w, output_w))

print("joint_log_prob test:", My_HMC_NN.joint_log_prob(initial_state))


samples = My_HMC_NN.hamiltonian_monte_carlo(10, initial_state)
