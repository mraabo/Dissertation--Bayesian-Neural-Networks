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

import imblearn as im
print(im.__version__)

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
credit_data = pd.read_csv("Python_code/data/UCI_Credit_Card.csv",encoding="utf-8",index_col=0, delimiter=",")
credit_data.head()
# Data to numpy
data=np.array(credit_data)
# seperating labels from features
data_X=data[:,0:23]
data_y=data[:,23]

data_X=data_X[0:100,:]
data_y=data_y[0:100]

# oversample = im.over_sampling.RandomOverSampler(sampling_strategy='minority')
# data_X, data_y = oversample.fit_resample(data_X, data_y)

# # ----------------------------- Subsamling credit data ---------------------------
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.30, random_state=3030)


# # ----------------------------- Implementing a BNN function ---------------------------
def construct_bnn(ann_input, ann_output, n_hidden):
    # Initialize random weights between each layer
    init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
    #init_2 = np.random.randn(n_hidden,n_hidden ).astype(floatX)
    init_out = np.random.randn(n_hidden,1).astype(floatX)

    with pm.Model() as bayesian_neural_network:
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", y_train)
        
        
        # prior on hyper parameters for weight 1
        #mu1 = pm.Normal('mu1',shape=(X_train.shape[1], n_hidden), mu=0, sigma=1)
        mu1 = pm.Cauchy('mu1',shape=(X_train.shape[1], n_hidden), alpha=0, beta=1)
        sigma1 = pm.HalfNormal('sigma1',shape=(X_train.shape[1], n_hidden), sigma=1)
        
        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", mu1, sigma1, shape=(X_train.shape[1], n_hidden), testval=init_1)

        # Weights from 1st to 2nd layer
        #weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden, n_hidden), testval=init_2)
        
        # prior on hyper parameters for weight_out
        mu_out = pm.Cauchy('mu_out',shape=(n_hidden, 1), alpha=0, beta=1)
        sigma_out = pm.HalfNormal('sigma_out',shape=(n_hidden, 1), sigma=1) 
        # Weights from hidden layer to output
        weights_1_out = pm.Normal("weights_out", mu_out, sigma=sigma_out, shape=(n_hidden,1), testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        #act_2 = pm.math.sigmoid(pm.math.dot(act_1, weights_1_2))
        #act_out = pm.Deterministic("act_out",pm.math.sigmoid(tt.dot(act_1, weights_1_out)))
        act_out = pm.math.sigmoid(tt.dot(act_1, weights_1_out))

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
    trace = pm.sample(draws=3000, tune=1000,chains=3,target_accept=.9)
    
# Making predictions using the posterior predective distribution
ppc1=pm.sample_posterior_predictive(trace,model=bayesian_neural_network_NUTS)

y_train_pred = ppc1['out'].mean(axis=0)
y_train_pred = (y_train_pred > 0.25).astype(int)
y_train_pred = y_train_pred[0] 

# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": X_test, "ann_output": y_test}, model=bayesian_neural_network_NUTS)
ppc2 = pm.sample_posterior_predictive(trace, model=bayesian_neural_network_NUTS)
y_test_pred = ppc2['out'].mean(axis=0)
y_test_pred=y_test_pred > 0.25
y_test_pred=y_test_pred[0]

# end time
toc = time.time()  
print(f"Running MCMC completed in {toc - tic:} seconds")

# Printing the performance measures
print('Accuracy on train data = {}%'.format(accuracy_score(y_train, y_train_pred) * 100))
print('Accuracy on test data = {}%'.format(accuracy_score(y_test, y_test_pred) * 100))

# Confusing matrix
cm=confusion_matrix(y_test,y_test_pred, normalize='all')
loss = cm[0,1]*10+cm[1,0]

sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)
plt.show()



# # Visualizing the trace
# fig, axes = plt.subplots(3,2, figsize=(12,6))
# with bayesian_neural_network_NUTS:
#     az.plot_trace(trace)
# fig.tight_layout()
# fig.show()
# print("yolo")

