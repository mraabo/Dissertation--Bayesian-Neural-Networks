import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano
floatX = theano.config.floatX
theano.config.mode = 'FAST_COMPILE'


def construct_nn(ann_input, ann_output):
    n_hidden = 5
    n_features = ann_input.get_value().shape[1]
    # Initialize random weights between each layer
    init_1 = np.random.randn(1, n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden, 1).astype(floatX)
    with pm.Model() as neural_network:

        weights_1 = pm.Normal('w_1', mu=0, shape=(
            n_features, n_hidden), testval=init_1)
        weights_2 = pm.Normal('w_2', mu=0, shape=(
            n_hidden, n_hidden), testval=init_2)
        weights_3 = pm.Normal('w_3', mu=0, shape=(
            n_hidden, 1), testval=init_out)

        # Activations
        act_1 = pm.math.tanh(pm.math.dot(ann_input, weights_1))
        act_2 = pm.math.tanh(pm.math.dot(act_1, weights_2))
        act_3 = pm.math.sigmoid(pm.math.dot(act_2, weights_3))
        out = pm.Normal('out', act_3, observed=ann_output)
    return neural_network


trainX = np.array([[1., 21., 51., 53]]).T
trainY = np.array([1, 1, 2, 2])

ann_input = theano.shared(np.array(trainX))
ann_output = theano.shared(np.array(trainY))
neural_network = construct_nn(ann_input, ann_output)


with neural_network:
    trace = pm.sample(50, tune=10, cores=1)

print(trace[0:10])

print("end")
