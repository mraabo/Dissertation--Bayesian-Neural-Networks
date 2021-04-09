#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:26:26 2021

@author: ulrikroed-sorensen
"""

from arviz.utils import Numba
Numba.disable_numba()
Numba.numba_flag
import numpy as np
import pymc3 as pm
import theano
import arviz as az
floatX = theano.config.floatX

# def construct_nn(ann_input, ann_output):
#     n_hidden = 5
#     n_features = ann_input.get_value().shape[1]
#     # Initialize random weights between each layer
#     init_1 = np.random.randn(1, n_hidden).astype(floatX)
#     init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
#     init_out = np.random.randn(n_hidden, 1).astype(floatX) 
#     with pm.Model() as neural_network:

#         weights_1 = pm.Normal('w_1', mu=0, shape=(
#             n_features, n_hidden), testval=init_1)
#         weights_2 = pm.Normal('w_2', mu=0, shape=(
#             n_hidden, n_hidden), testval=init_2)
#         weights_3 = pm.Normal('w_3', mu=0, shape=(
#             n_hidden, 1), testval=init_out)

#         # Activations
#         act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_1))
#         act_2 = pm.math.tanh(pm.math.dot(act_1, weights_2))
#         act_3 = pm.math.sigmoid(pm.math.dot(act_2, weights_3))
#         #out = pm.Normal('out', act_3, observed=ann_output)
#         out = pm.Bernoulli(
#             "out",
#             act_3, observed=ann_output)
    
#     return neural_network



def construct_nn(ann_input, ann_output):
    n_hidden = 5

    # Initialize random weights between each layer
    init_1 = np.random.randn(trainX.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden).astype(floatX)

    with pm.Model() as neural_network:
        # Trick: Turn inputs and outputs into shared variables using the data container pm.Data
        # It's still the same thing, but we can later change the values of the shared variable
        # (to switch in the test-data later) and pymc3 will just use the new data.
        # Kind-of like a pointer we can redirect.
        # For more info, see: http://deeplearning.net/software/theano/library/compile/shared.html
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

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
            "out",
            act_out,
            observed=ann_output,
            total_size=trainY.shape[0],  # IMPORTANT for minibatches
        )
    return neural_network




trainX = np.array([[1., 2.],[1., 2.],[ 51., 53],[ 51., 53]])
trainY = np.array([0, 0, 1, 1])

# ann_input = theano.shared(np.array(trainX))
# ann_output = theano.shared(np.array(trainY))
neural_network = construct_nn(trainX, trainY)


with neural_network:
    trace = pm.sample(draws=5000, tune=1000, cores=2, chains=1)
    
#print(np.array(trace))
with neural_network:
   az.plot_trace(trace)
    

prediction=pm.sample_posterior_predictive(trace, model=neural_network)
pred_mean=np.mean(prediction['out'], axis=0)

print(prediction['out'].shape)






print("end")