#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 11 13:32:53 2021

@author: ulrikroed-sorensen
"""
# ------------------- Imports for BNN PYMC3  --------------------------------- 
import numpy as np
import pymc3 as pm

import theano
import arviz as az
from arviz.utils import Numba
from scipy.stats import mode
import theano.tensor as tt
Numba.disable_numba()
Numba.numba_flag
floatX = theano.config.floatX
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def construct_nn(ann_input, ann_output, n_hidden = 5, task="regression"):
    # Initialize random weights between each layer
    init_1 = np.random.randn(trainX.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as neural_network:
        ann_input = pm.Data("ann_input", trainX)
        ann_output = pm.Data("ann_output", trainY)

        # Weights from input to hidden layer
        weights_in_1 = pm.Normal("w_in_1", 0, sigma=1, shape=(trainX.shape[1], n_hidden), testval=init_1)

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal("w_1_2", 0, sigma=1, shape=(n_hidden, n_hidden), testval=init_2)

        # Weights from hidden layer to output
        weights_2_out = pm.Normal("w_2_out", 0, sigma=1, shape=(n_hidden,), testval=init_out)

        # Build neural-network using tanh activation function
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_in_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_1_2))
        act_out = pm.math.sigmoid(pm.math.dot(act_2, weights_2_out))
        if task == "regression":
            # Normal likelihood for regression task
            out = pm.Normal('out', act_out, observed=ann_output)
   
        elif task == "classification":     
            #Binary classification -> Bernoulli likelihood
                 # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli(
                "out",
                act_out,
                observed=ann_output,
                total_size=trainY.shape[0],  # IMPORTANT for minibatches
            )
            
        elif task == "multinomial":
            act_1 = pm.Deterministic('activations_1',
                              tt.tanh(tt.dot(ann_input, weights_in_1)))
            act_2 = pm.Deterministic('activations_2',
                              tt.tanh(tt.dot(act_1, weights_1_2)))
            act_out = pm.Deterministic('activations_out',
                                tt.nnet.softmax(tt.dot(act_2, weights_2_out))) 
            act_out = tt.nnet.softmax(pm.math.dot(act_2, weights_2_out))
            
            out = pm.Categorical('out',
                        act_out,
                        observed = ann_output)
            
    return neural_network

# # ----------------------------- Bank data load ---------------------------

# Importing traning data set
data=np.genfromtxt("data_banknote_authentication.txt", delimiter = ",")

# reshaping to form a 784 X 10000 matrix
dataX=data[:,0:4]
dataY=data[:,4]

# Splitting into train and test
trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=0.5)




# # ----------------------------- Making predicitions ---------------------------

# Constructing af NN
neural_network = construct_nn(trainX, trainY, n_hidden=10,task="classification")
# Sample from the posterior using the NUTS samplper
with neural_network:
    trace = pm.sample(draws=5000, tune=1000, cores=2, chains=1)
    

# Visualizing the trace
with neural_network:
   az.plot_trace(trace)
   
# with neural_network:
#     inference = pm.ADVI()  # approximate inference done using ADVI
#     approx = pm.fit(10000, method=inference)
#     trace = approx.sample(500)
       
   
# Making predictions using the posterior predective distribution
prediction=pm.sample_posterior_predictive(trace, model=neural_network)

# Relative frequency of predicting class 1
pred = prediction['out'].mean(axis=0)


# Returns the most common value in array (majority vote)
y_pred = mode(prediction['out'], axis=0).mode[0, :]


# Accuracy
print('Accuracy on train data = {}%'.format(accuracy_score(trainY, y_pred) * 100))


# Probability surface
# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": testX, "ann_output": testY}, model=neural_network)

# Creater posterior predictive samples
ppc = pm.sample_posterior_predictive(trace, model=neural_network, samples=500)

# Returns the most common value in array (majority vote)
pred= mode(ppc['out'], axis=0).mode[0, :]


print('Accuracy on test data = {}%'.format((testY == pred).mean() * 100))
