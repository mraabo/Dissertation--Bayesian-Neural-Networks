# # ----------------------------- INFO ---------------------------
# In this python script we implement and run a BNN for predicting default on
# credit cards. The sampler is based on the NUTS sampler

# # ----------------------------- IMPORTS ---------------------------
import sys
import time
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import pymc3 as pm
import theano
import arviz as az
from arviz.utils import Numba
from scipy.stats import mode
Numba.disable_numba()
Numba.numba_flag
floatX = theano.config.floatX
import seaborn as sns
sns.set_style("white")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# # ----------------------------- Print versions ---------------------------
print("Running on Python version %s" % sys.version)
print(f"Running on PyMC3 version{pm.__version__}")
print("Running on Theano version %s" % theano.__version__)
print("Running on Arviz version %s" % az.__version__)
print("Running on Numpy version %s" % np.__version__)

# # ----------------------------- Loading credit data ---------------------------

credit_AU_data = np.genfromtxt('Python_code/data/australian.dat')

# Data to numpy
data = np.array(credit_AU_data)

# Extract labels
data_X = data[:, 0:14]
data_y = data[:, 14]

X_train, X_test, y_train, y_test = train_test_split( data_X, data_y, test_size=0.30, random_state=42)


# # ----------------------------- Implementing a BNN function ---------------------------
def construct_bnn(ann_input, ann_output, n_hidden = 5):
    # Initialize random weights between each layer
    init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as bayesian_neural_network:
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", y_train)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(X_train.shape[1], n_hidden), testval=init_1)

        # Weights from hidden layer to output
        weights_1_out = pm.Normal("w_1_out", 0, sigma=1, shape=(n_hidden), testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        act_out = pm.math.sigmoid(pm.math.dot(act_1, weights_1_out))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
                "out",
                act_out,
                observed=ann_output,
                total_size=y_train.shape[0],  # IMPORTANT for minibatches
            )
            
    return bayesian_neural_network


# # ----------------------------- Sampling from posterior ---------------------------
tic = time.perf_counter() # for timing
bayesian_neural_network_NUTS = construct_bnn(X_train, y_train, n_hidden=10)

# Sample from the posterior using the NUTS samplper
with bayesian_neural_network_NUTS:
    trace = pm.sample(draws=3000, tune=1000, chains=3, target_accept=.90)
       
# Making predictions using the posterior predective distribution
ppc1=pm.sample_posterior_predictive(trace, model=bayesian_neural_network_NUTS)

# Returns the most common value in array (majority vote)
y_train_pred = mode(ppc1['out'], axis=0).mode[0, :]

# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": X_test, "ann_output": y_test}, model=bayesian_neural_network_NUTS)
ppc2 = pm.sample_posterior_predictive(trace, model=bayesian_neural_network_NUTS)

# Returns the most common value in array (majority vote)
y_test_pred = mode(ppc2['out'], axis=0).mode[0, :]

# End time
toc = time.perf_counter()
print(f"Run time {toc - tic:0.4f} seconds")

# Printing the performance measures
print('Accuracy on train data = {}%'.format(accuracy_score(y_train, y_train_pred) * 100))
print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_test_pred) * 100))

# Confusing matrix
cm=confusion_matrix(y_test_pred,y_test, normalize='all')
sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)
plt.show()

# Visualizing the trace
with bayesian_neural_network_NUTS:
    az.plot_trace(trace)

# # ----------------------------- UNDER CONSTRUCTION ;-) ---------------------------
### DO NOT DELELTE THIS YET

# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# pred3=clf.predict(X_train)


# print('Accuracy on train data = {}%'.format(accuracy_score(y_train, pred3) * 100))

