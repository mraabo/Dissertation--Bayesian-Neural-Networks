# # ----------------------------- INFO ---------------------------
# In this python script we implement and run a BNN for predicting house prices
# in Boston. The sampler is based on the NUTS sampler

# # ----------------------------- IMPORTS ---------------------------
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time
from keras.datasets import boston_housing
from sklearn import metrics
import numpy as np
import pymc3 as pm
import theano
import arviz as az
from arviz.utils import Numba
import theano.tensor as tt
Numba.disable_numba()
Numba.numba_flag
floatX = theano.config.floatX
sns.set_style("white")

# # ----------------------------- Print versions ---------------------------

print("Running on Python version %s" % sys.version)
print(f"Running on PyMC3 version{pm.__version__}")
print("Running on Theano version %s" % theano.__version__)
print("Running on Arviz version %s" % az.__version__)
print("Running on Numpy version %s" % np.__version__)

# Ignore warnings - NUTS provide many runtimeWarning
warnings.filterwarnings("ignore", category=RuntimeWarning)

tf.random.set_seed(42)
# # ----------------------------- Loading Boston data ---------------------------
(X_train, y_train), (X_test, y_test) = boston_housing.load_data(seed=3030)

# pad Xs with 1's to add bias
ones_train = np.ones(X_train.shape[0])
ones_test = np.ones(X_test.shape[0])
X_train = np.insert(X_train, 0, ones_train, axis=1)
X_test = np.insert(X_test, 0, ones_test, axis=1)



# # ----------------------------- Implementing a BNN function ---------------------------


def construct_bnn(ann_input, ann_output, n_hidden, prior_std):
    # Initialize random weights between each layer
    init_out = np.random.randn(X_train.shape[1], 1).astype(floatX)*prior_std

    with pm.Model() as bayesian_neural_network:
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", y_train)

    # Input layer -> Output layer
        weights_out = pm.Normal('w_out', mu=0, sd=prior_std,
                                shape=(X_train.shape[1], 1),
                                testval=init_out)
        acts_out = pm.Deterministic(
            'activations_out', tt.dot(ann_input, weights_out))

    # Define likelihood
        out = pm.Normal('out', mu=acts_out[:, 0], sd=1, observed=ann_output)

    return bayesian_neural_network


# # ----------------------------- Sampling from posterior ---------------------------

# Start time
tic = time.perf_counter()  # for timing
bayesian_neural_network_NUTS = construct_bnn(
    X_train, y_train, n_hidden=10, prior_std=0.1)

# Sample from the posterior using the NUTS samplper
with bayesian_neural_network_NUTS:
    trace = pm.sample(draws=3000, tune=1000, chains=3, target_accept=.9)


# # ----------------------------- Making predictions on training data ---------------------------
ppc1 = pm.sample_posterior_predictive(
    trace, model=bayesian_neural_network_NUTS)

# Taking the mean over all samples to generate a prediction
y_train_pred = ppc1['out'].mean(axis=0)


# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": X_test, "ann_output": y_test},
            model=bayesian_neural_network_NUTS)


# # ----------------------------- Making predictions on test data ---------------------------
ppc2 = pm.sample_posterior_predictive(
    trace, model=bayesian_neural_network_NUTS)

# Taking the mean over all samples to generate a prediction
y_test_pred = ppc2['out'].mean(axis=0)

# End time
toc = time.perf_counter()
print(f"Run time {toc - tic:0.4f} seconds")

# Printing the performance measures
print('MSE (NUTS) on training data:',
      metrics.mean_squared_error(y_train, y_train_pred))
print('MSE (NUTS) on test data:', metrics.mean_squared_error(y_test, y_test_pred))



# Vizualize uncertainty by plotting posterior predictive
ppc2_array = np.array(ppc2['out'])
plt.hist(ppc2_array[:, 40], density=1, color="skyblue")
plt.show()

