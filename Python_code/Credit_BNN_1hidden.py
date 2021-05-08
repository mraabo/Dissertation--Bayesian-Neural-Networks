# # ----------------------------- INFO ---------------------------
# In this python script we implement and run a BNN for predicting default on
# credit card clients. The sampler is based on the NUTS sampler

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
import theano.tensor as tt
from scipy.stats import mode
Numba.disable_numba()
Numba.numba_flag
floatX = theano.config.floatX
import tensorflow as tf
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

# Ignore warnings - NUTS provide many runtimeWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

tf.random.set_seed(42)


# # ----------------------------- Loading credit data ---------------------------

credit_data = pd.read_csv("UCI_Credit_Card.csv",encoding="utf-8",index_col=0, delimiter=";")
credit_data.head()
# Data to numpy
data=np.array(credit_data)
# seperating labels from features
data_X=data[:,0:23]
data_y=data[:,23]

data_X=data_X[0:1000,:]
data_y=data_y[0:1000]

# # ----------------------------- Subsamling credit data ---------------------------
X_train, X_test, y_train, y_test = train_test_split( data_X, data_y, test_size=0.30, random_state=3030)


# # ----------------------------- Implementing a BNN function ---------------------------
def construct_bnn(ann_input, ann_output, n_hidden = 5, prior_std=10):
    # Initialize random weights between each layer
    init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)*prior_std
    #init_2 = np.random.randn(n_hidden,n_hidden ).astype(floatX)
    init_out = np.random.randn(n_hidden,1).astype(floatX)*prior_std

    with pm.Model() as bayesian_neural_network:
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", y_train)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=prior_std, shape=(X_train.shape[1], n_hidden), testval=init_1)

        # Weights from 1st to 2nd layer
        #weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden, n_hidden), testval=init_2)

        # Weights from hidden layer to output
        weights_1_out = pm.Normal("weights_out", 0, sigma=prior_std, shape=(n_hidden,1), testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        #act_2 = pm.math.sigmoid(pm.math.dot(act_1, weights_1_2))
        act_out = pm.Deterministic("act_out",pm.math.sigmoid(tt.dot(act_1, weights_1_out)))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
                "out",
                act_out,
                observed=ann_output,
                total_size=y_train.shape[0],  # IMPORTANT for minibatches
            )
            
    return bayesian_neural_network


# # ----------------------------- Sampling from posterior ---------------------------
tic = time.time() # for timing
bayesian_neural_network_NUTS = construct_bnn(X_train, y_train, n_hidden=10)

# Sample from the posterior using the NUTS samplper
with bayesian_neural_network_NUTS:
    trace = pm.sample(draws=3000, tune=2000,chains=1)
    
# end time
toc = time.time()  
print(f"Running MCMC completed in {toc - tic:} seconds")

# Making predictions using the posterior predective distribution
ppc1=pm.sample_posterior_predictive(trace, var_names=["act_out"],model=bayesian_neural_network_NUTS)

y_train_pred=mode(ppc1['act_out']>0.25,axis=0).mode[0:].astype(int)
y_train_pred=y_train_pred.reshape(y_train.shape[0],1)


# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": X_test, "ann_output": y_test}, model=bayesian_neural_network_NUTS)
ppc2 = pm.sample_posterior_predictive(trace,var_names=["act_out"], model=bayesian_neural_network_NUTS, samples=50)
y_test_pred=mode(ppc2['act_out']>0.25,axis=0).mode[0:,].astype(int)
y_test_pred=y_test_pred.reshape(y_test.shape[0],1)

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
    
    
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.predict(X_train)

prob=clf.predict_proba(X_train)

prob[:,1]

print(clf.coef_, clf.intercept_)

cm=confusion_matrix(clf.predict(X_test),y_test, normalize='all')
sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)
plt.show()
