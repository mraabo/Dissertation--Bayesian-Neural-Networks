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
from sklearn.metrics import confusion_matrix, accuracy_score, log_loss


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
credit_data = pd.read_csv("Python_code/data/UCI_Credit_Card.csv",encoding="utf-8",index_col=0, delimiter=",")
credit_data.head()
# Data to numpy
data=np.array(credit_data)
# seperating labels from features
data_X=data[:,0:23]
data_y=data[:,23]


# # ----------------------------- Subsamling credit data ---------------------------
X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.30, random_state=3030)

N = 300
N_test = 100
X_train=X_train[0:N,:]
y_train=y_train[0:N]
X_test=X_test[0:N_test,:]
y_test=y_test[0:N_test]


# pad Xs with 1's to add bias
ones_train = np.ones(X_train.shape[0])
ones_test = np.ones(X_test.shape[0])
X_train = np.insert(X_train, 0, ones_train, axis=1)
X_test = np.insert(X_test, 0, ones_test, axis=1)


# # ----------------------------- Implementing a BNN function ---------------------------
def construct_bnn(ann_input, ann_output, n_hidden, prior_std):
    # # Initialize random weights between each layer
    # init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)*prior_std
    # #init_2 = np.random.randn(n_hidden,n_hidden ).astype(floatX)
    # init_out = np.random.randn(n_hidden,1).astype(floatX)*prior_std

    with pm.Model() as bayesian_neural_network:
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", y_train)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=prior_std, shape=(X_train.shape[1], n_hidden))

        # Weights from 1st to 2nd layer
        #weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden, n_hidden), testval=init_2)

        # Weights from hidden layer to output
        weights_1_out = pm.Normal("weights_out", 0, sigma=prior_std, shape=(n_hidden,1))

        # Build neural-network using tanh activation function
        act_1 =  pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        #act_2 = pm.math.sigmoid(pm.math.dot(act_1, weights_1_2))
        #act_out = pm.Deterministic("act_out",pm.math.sigmoid(tt.dot(act_1, weights_1_out)))
        output = pm.Deterministic("output",pm.math.sigmoid(tt.dot(act_1, weights_1_out)))

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
                "out",
                output,
                observed=ann_output,
                total_size=y_train.shape[0],  # IMPORTANT for minibatches
            )
            
    return bayesian_neural_network


# # ----------------------------- Sampling from posterior ---------------------------
tic = time.time() # for timing
bayesian_neural_network_NUTS = construct_bnn(X_train, y_train, n_hidden=10,prior_std=1)

# Sample from the posterior using the NUTS samplper
draws=1500
tune=10**3
chains=3
target_accept=.9
with bayesian_neural_network_NUTS:
    trace = pm.sample(draws=draws, tune=tune,chains=chains,target_accept=target_accept)


y_train_pred = (trace["output"]).mean(axis=0)

# Making predictions using the posterior predective distribution
# ppc1=pm.sample_posterior_predictive(trace,model=bayesian_neural_network_NUTS)

# y_train_pred = ppc1['out'].mean(axis=0)
# y_train_pred = (y_train_pred > 0.5).astype(int)
# y_train_pred = y_train_pred[0] 

# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": X_test, "ann_output": y_test}, model=bayesian_neural_network_NUTS)

ppc2 = pm.sample_posterior_predictive(trace,var_names=["output"], model=bayesian_neural_network_NUTS)
y_test_pred = (ppc2["output"]).mean(axis=0)

# y_test_pred = np.append(y_test_pred,1-y_test_pred,axis=1)

# end time
toc = time.time()  
print(f"Running MCMC completed in {toc - tic:} seconds")

# Printing the performance measures
print('Cross-entropy loss on train data = {}'.format(log_loss(y_train, y_train_pred)))
print('Cross-entropy loss on test data = {}'.format(log_loss(y_test, y_test_pred)))

# Confusing matrix
test_class_pred = y_test_pred[:,0]>0.5
cm=confusion_matrix(y_test,test_class_pred, normalize='all')
sns.heatmap(cm, cmap=plt.cm.Blues, annot=True)
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.show()
plt.savefig('Python_code/Credit_BNN_1hidden_confusion_matrix.pdf')

# Vizualize uncertainty
# Define examples for which you want to examine the posterior predictive:
example_vec=np.array([4,11,25,27,33,44,58,88,104])
for example in example_vec:
    plt_hist_array=np.array(ppc2['output'])
    plt.hist(plt_hist_array[:,example], density=True, color="lightsteelblue", bins=30)
    plt.xlabel(f"Predicted probability for example {example}")
    plt.ylabel("Relative frequency")
    plt.savefig(f'Python_code/Credit_BNN_1hidden_postpred_{example}.pdf')
    plt.show()
