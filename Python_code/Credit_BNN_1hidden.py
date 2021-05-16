# # ----------------------------- INFO ---------------------------
# In this python script we implement and run a BNN for predicting default on
# credit card clients. The sampler is based on the NUTS sampler

# # ----------------------------- IMPORTS ---------------------------
import warnings
from sklearn.metrics import accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import sys
import time
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import pymc3 as pm
import theano
import arviz as az
from arviz.utils import Numba
import theano.tensor as tt
from scipy.stats import mode
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


# # ----------------------------- Loading credit data ---------------------------
credit_data = pd.read_csv("Python_code/data/UCI_Credit_Card.csv",
                          encoding="utf-8", index_col=0, delimiter=",")
credit_data.head()
# Data to numpy
data = np.array(credit_data)
# seperating labels from features
data_X = data[:, 0:23]
data_y = data[:, 23]


# # ----------------------------- Subsamling credit data ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data_X, data_y, test_size=0.30, random_state=3030)

N = 300
N_test = 100
X_train = X_train[0:N, :]
y_train = y_train[0:N]
X_test = X_test[0:N_test, :]
y_test = y_test[0:N_test]


# pad Xs with 1's to add bias
ones_train = np.ones(X_train.shape[0])
ones_test = np.ones(X_test.shape[0])
X_train = np.insert(X_train, 0, ones_train, axis=1)
X_test = np.insert(X_test, 0, ones_test, axis=1)


# # ----------------------------- Implementing a BNN function ---------------------------
def construct_bnn(ann_input, ann_output, n_hidden, prior_std):

    with pm.Model() as bayesian_neural_network:
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", y_train)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal(
            "w_in_1", 0, sigma=prior_std, shape=(X_train.shape[1], n_hidden))

        # Weights from hidden layer to output
        weights_1_out = pm.Normal(
            "weights_out", 0, sigma=prior_std, shape=(n_hidden, 1))

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        output = pm.Deterministic(
            "output", pm.math.sigmoid(tt.dot(act_1, weights_1_out)))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
            "out",
            output,
            observed=ann_output,
            total_size=y_train.shape[0],  # IMPORTANT for minibatches
        )

    return bayesian_neural_network


# # ----------------------------- Sampling from posterior ---------------------------
tic = time.time()  # for timing
bayesian_neural_network_NUTS = construct_bnn(
    X_train, y_train, n_hidden=10, prior_std=1)

# Sample from the posterior using the NUTS samplper
draws = 1500
tune = 10**3
chains = 3
target_accept = .9
with bayesian_neural_network_NUTS:
    trace = pm.sample(draws=draws, tune=tune, chains=chains,
                      target_accept=target_accept)


y_train_pred = (trace["output"]).mean(axis=0)

# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": X_test, "ann_output": y_test},
            model=bayesian_neural_network_NUTS)

ppc2 = pm.sample_posterior_predictive(
    trace, var_names=["output"], model=bayesian_neural_network_NUTS)
y_test_pred = (ppc2["output"]).mean(axis=0)


# end time
toc = time.time()
print(f"Running MCMC completed in {toc - tic:} seconds")

# Printing the performance measures
print('Cross-entropy loss on train data = {}'.format(log_loss(y_train, y_train_pred)))
print('Cross-entropy loss on test data = {}'.format(log_loss(y_test, y_test_pred)))


# Vizualize uncertainty
# Define examples for which you want to examine the posterior predictive:
example_vec = np.array([5, 11, 25, 88])
for example in example_vec:
    plt_hist_array = np.array(ppc2['output'])
    plt.hist(plt_hist_array[:, example], density=1,
             color="lightsteelblue", bins=30)
    plt.xlabel(f"Predicted probability for example {example}", fontsize=13)
    plt.ylabel("Density", fontsize=13)
    #plt.savefig(f'Python_code/Credit_BNN_1hidden_postpred_{example}.pdf')
    plt.show()
