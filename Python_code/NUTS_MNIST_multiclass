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

# For creating toy data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set_style("white")

# # ----------------------------- MNIST data load ---------------------------

# Importing traning data set
trainX_clean=np.genfromtxt("MNIST-Train-cropped.txt")
# reshaping to form a 784 X 10000 matrix
trainX_clean=trainX_clean.reshape(784,10000, order="F")

#T Importing traning labels
trainY_clean=np.genfromtxt("MNIST-Train-Labels-cropped.txt")
# Importing test data set
test_data_clean=np.genfromtxt("MNIST-Test-cropped.txt")
# reshaping to form a 784 X 2000 matrix
test_data_clean=test_data_clean.reshape(784,2000, order = "F")
#Importing test labels
test_labels_clean=np.genfromtxt("MNIST-Test-Labels-cropped.txt")


# plot images
num_row = 6 # number of rows in plot
num_col = 6 # number of columns in plot
fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(0,36):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(trainX_clean[:,i].reshape(28,28,order="F"), cmap='Blues')
    ax.set_title('Label: {}'.format(trainY_clean[i]))
plt.tight_layout()
plt.show()

# Making the sample size smaller
trainX=trainX_clean[:,0:50]

trainY=trainY_clean[0:50]
testX=test_data_clean[:,0:50]
testY=test_labels_clean[0:50]

# Tranposing training data in order to run through our NN model
trainX=trainX.T
test=testX.T

# ------------------- Defining a BNN function  --------------------------------- 
def construct_nn(ann_input, ann_output, n_hidden = 5):
    # Initialize random weights between each layer
    init_1 = np.random.randn(trainX.shape[1], n_hidden).astype(floatX) 
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden, trainY.shape[0]).astype(floatX)

    with pm.Model() as neural_network:
        ann_input = pm.Data("ann_input", trainX)
        ann_output = pm.Data("ann_output", trainY)
        
    # Input -> Layer 1
        weights_1 = pm.Normal('w_1', mu=0, sd=1,
                          shape=(trainX.shape[1], n_hidden),
                          testval=init_1)
        acts_1 = pm.Deterministic('activations_1', tt.tanh(tt.dot(ann_input, weights_1)))
        
    # Layer 1 -> Layer 2
        weights_2 = pm.Normal('w_2', mu=0, sd=1,
                          shape=(n_hidden, n_hidden),
                          testval=init_2)
        acts_2 = pm.Deterministic('activations_2', tt.tanh(tt.dot(acts_1, weights_2)))

    # Layer 2 -> Output Layer
        weights_out = pm.Normal('w_out', mu=0, sd=1,
                            shape=(n_hidden, trainY.shape[0]),
                            testval=init_out)
        acts_out = pm.Deterministic('activations_out',tt.nnet.softmax(tt.dot(acts_2, weights_out)))

    # Define likelihood
        out = pm.Multinomial('likelihood', n=1, p=acts_out,
                          observed=ann_output)

        
        
    return neural_network


# # ----------------------------- Making predicitions ---------------------------

# Constructing af NN
neural_network = construct_nn(trainX, trainY, n_hidden=20)
# Sample from the posterior using the NUTS samplper
with neural_network:
    trace = pm.sample(draws=5000, tune=1000, cores=2, chains=1)
    

# Visualizing the trace
with neural_network:
   az.plot_trace(trace)
   
with neural_network:
    inference = pm.ADVI()  # approximate inference done using ADVI
    approx = pm.fit(10000, method=inference)
    trace = approx.sample(500)
    

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

# Use probability of > 0.5 to assume prediction of class 1
pred = ppc['out'].mean(axis=0) > 0.5


print('Accuracy on test data = {}%'.format((testY == pred).mean() * 100))

