# ------------------- Imports for BNN PYMC3  --------------------------------- 
import numpy as np
import pymc3 as pm
import theano
import arviz as az
from arviz.utils import Numba
from scipy.stats import mode
Numba.disable_numba()
Numba.numba_flag
floatX = theano.config.floatX

# For creating toy data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set_style("white")

# For setting a seed
from pymc3.theanof import MRG_RandomStreams, set_tt_rng

# ------------------- Neural Network function --------------------------------- 
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
            # Binary classification -> Bernoulli likelihood
            out = pm.Bernoulli(
                "out",
                act_out,
                observed=ann_output,
                total_size=trainY.shape[0],  # IMPORTANT for minibatches
            )
            
    return neural_network


# ----------------------------- Toy data example -----------------------------
    

# Moons data
X, Y = make_moons(noise=0.2, random_state=0, n_samples=1000)
X = scale(X)
X = X.astype(floatX)
Y = Y.astype(floatX)
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.5)

# Visualize data
fig, ax = plt.subplots()
ax.scatter(X[Y == 0, 0], X[Y == 0, 1], label="Class 0",color='b',edgecolors='k',alpha=0.6)
ax.scatter(X[Y == 1, 0], X[Y == 1, 1], color="r", label="Class 1",edgecolors='k',alpha=0.6)
sns.despine()
ax.legend()
ax.set(xlabel="X1", ylabel="X2", title="Toy binary classification data set");


# set seed to 42
set_tt_rng(MRG_RandomStreams(42))

neural_network = construct_nn(trainX, trainY,task="classification")

# Sample from the posterior using the NUTS samplper
with neural_network:
    trace = pm.sample(draws=5000, tune=1000, cores=2, chains=1)
    

# Visualizing the trace
with neural_network:
   az.plot_trace(trace)
    

# Making predictions using the posterior predective distribution
prediction=pm.sample_posterior_predictive(trace, model=neural_network)

# Relative frequency of predicting class 1
pred = prediction['out'].mean(axis=0)

# Returns the most common value in array (majority vote)
y_pred = mode(prediction['out'], axis=0).mode[0, :]

# HeatMap:
sns.heatmap(confusion_matrix(trainY,y_pred)/500)

# Accuracy
print('Accuracy on train data = {}%'.format(accuracy_score(trainY, y_pred) * 100))


# Probability surface
# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": testX, "ann_output": testY}, model=neural_network)

# Creater posterior predictive samples
ppc = pm.sample_posterior_predictive(trace, model=neural_network, samples=500)

# Use probability of > 0.5 to assume prediction of class 1
pred = ppc['out'].mean(axis=0) > 0.5

fig, ax = plt.subplots()
ax.scatter(testX[pred==0, 0], testX[pred==0, 1], color='b',edgecolors='k',alpha=0.6)
ax.scatter(testX[pred==1, 0], testX[pred==1, 1], color='r',edgecolors='k',alpha=0.6)
sns.despine()
ax.set(title='Predicted labels in testing set', xlabel='X1', ylabel='X2');

print('Accuracy on test data = {}%'.format((testY == pred).mean() * 100))
