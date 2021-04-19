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

# For creating toy data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
sns.set_style("white")



# ------------------- Defining a NN function  --------------------------------- 
def construct_nn(ann_input, ann_output, n_hidden = 5):
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

        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli(
                "out",
                act_out,
                observed=ann_output,
                total_size=trainY.shape[0],  # IMPORTANT for minibatches
            )
            
    return neural_network

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



# # ----------------------------- MNIST  binary classification ---------------
# Making the sample size smaller
trainX=trainX_clean[:,0:50]
trainY=trainY_clean[0:50]
test_data=test_data_clean[:,0:50]
test_labels=test_labels_clean[0:50]

# Fixing data for binary classification 
# First we need to select the two digigts for binary classification
num1 = 0
num2 = 8

# Indexing data to only include the numbers choosen for the binary classification
index = np.array(np.where((trainY != num1) & (trainY != num2))) 
    
# Defining af new traning set that only contains num1 and num2    
trainX = np.delete(trainX,index, axis=1) 
trainY = trainY[(trainY == num1) | (trainY == num2)] 

# set labels to 0 or 1 for binary classification
trainY=np.where(trainY == num1,0,1)

######  Filtering test data   ##########
# Test data set
index2 = np.array(np.where((test_labels != num1) & (test_labels != num2))) #
    
testX = np.delete(test_data,index2, axis=1) # Deletes all other than the two numbers selected
testY = test_labels[(test_labels == num1) | (test_labels == num2)] # only the selected labels
testY=np.where(testY == num1,0,1)


# # ----------------------------- Making predicitions ---------------------------

# Tranposing training data in order to run through our NN model
trainX=trainX.T
# Constructing af NN
neural_network = construct_nn(trainX, trainY, n_hidden=10)
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
pm.set_data(new_data={"ann_input": testX.T, "ann_output": testY}, model=neural_network)

# Creater posterior predictive samples
ppc = pm.sample_posterior_predictive(trace, model=neural_network, samples=500)

# Use probability of > 0.5 to assume prediction of class 1
pred = ppc['out'].mean(axis=0) > 0.5


print('Accuracy on test data = {}%'.format((testY == pred).mean() * 100))


