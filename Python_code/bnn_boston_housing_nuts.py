# # ----------------------------- INFO ---------------------------
# In this python script we implement and run a BNN for predicting house prices 
# in Boston. The sampler is based on the NUTS sampler

# # ----------------------------- IMPORTS ---------------------------
import sys
import time
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np 
import pymc3 as pm
import theano
import arviz as az
from arviz.utils import Numba
import theano.tensor as tt
Numba.disable_numba()
Numba.numba_flag
floatX = theano.config.floatX
# seaborn for vizualzing 
import seaborn as sns
sns.set_style("white")
# import pandas as pd
# from scipy.stats import norm
# from scipy.stats import t
# import matplotlib.pyplot as plt
# # ----------------------------- Print versions ---------------------------

print("Running on Python version %s" % sys.version)
print(f"Running on PyMC3 version{pm.__version__}")
print("Running on Theano version %s" % theano.__version__)
print("Running on Arviz version %s" % az.__version__)
print("Running on Numpy version %s" % np.__version__)



# # ----------------------------- Loading Boston data ---------------------------
# ones=np.ones(506)
X, y = load_boston(return_X_y=True)

#### DO NOT DELETE THIS YET ###

# boston_dataset = load_boston(StandardScaler=True)
# boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
# boston['MEDV'] = boston_dataset.target
# sns.set(rc={'figure.figsize':(11.7,8.27)})
# ax = sns.distplot(boston['MEDV'], kde = False, norm_hist=True) 
# ax = sns.distplot(np.random.normal(loc=22.0, scale=9.0, size=1000000), kde = True, hist=False) 
# plt.show()
# # X=np.insert(X,0,ones,axis=1)


# Splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# scaler=sc.fit(X_train)
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

X_train=X_train[0:100,:]
y_train=y_train[0:100,]
X_test=X_test[0:100,:]
y_test=y_test[0:100,]

# # ----------------------------- Implementing a BNN function ---------------------------

def construct_bnn(ann_input, ann_output, n_hidden = 5):
    # Initialize random weights between each layer
    init_1 = np.random.randn(X_train.shape[1], n_hidden).astype(floatX)
    init_2 = np.random.randn(n_hidden, n_hidden).astype(floatX)
    init_out = np.random.randn(n_hidden,1).astype(floatX)

    with pm.Model() as bayesian_neural_network:
        ann_input = pm.Data("ann_input", X_train)
        ann_output = pm.Data("ann_output", y_train)
        
    # Input -> Layer 1
        weights_1 = pm.Normal('w_1', mu=0, sd=1,
                          shape=(X_train.shape[1], n_hidden),
                          testval=init_1)
        acts_1 = pm.Deterministic('activations_1', tt.nnet.relu(tt.dot(ann_input, weights_1)))
        
    # Layer 1 -> Layer 2
        weights_2 = pm.Normal('w_2', mu=0, sd=1,
                          shape=(n_hidden, n_hidden),
                          testval=init_2)
        acts_2 = pm.Deterministic('activations_2', tt.nnet.relu(tt.dot(acts_1, weights_2)))

    # Layer 2 -> Output Layer
        weights_out = pm.Normal('w_out', mu=0, sd=1,
                            shape=(n_hidden,1),
                            testval=init_out)
        acts_out = pm.Deterministic('activations_out',tt.dot(acts_2, weights_out))
        

    #Define likelihood
        out = pm.Normal('out', mu=acts_out[:,0], sd=1, observed=ann_output)        
            
    return bayesian_neural_network


# # ----------------------------- Sampling from posterior ---------------------------
tic = time.perf_counter() # for timing
bayesian_neural_network_NUTS = construct_bnn(X_train, y_train, n_hidden=5)

# Sample from the posterior using the NUTS samplper
with bayesian_neural_network_NUTS:
    trace = pm.sample(draws=500, tune=100, cores=4, chains=1)
    

# Visualizing the trace
with bayesian_neural_network_NUTS:
    az.plot_trace(trace)

# # ----------------------------- Making predictions on training data ---------------------------
ppc1=pm.sample_posterior_predictive(trace, model=bayesian_neural_network_NUTS)

# Taking the mean over all samples to generate a prediction
y_train_pred = ppc1['out'].mean(axis=0)


# Replace shared variables with testing set
pm.set_data(new_data={"ann_input": X_test, "ann_output": y_test}, model=bayesian_neural_network_NUTS)

# # ----------------------------- Making predictions on test data ---------------------------
ppc2 = pm.sample_posterior_predictive(trace, model=bayesian_neural_network_NUTS, samples=1000)

# Taking the mean over all samples to generate a prediction
y_test_pred = ppc2['out'].mean(axis=0)

toc = time.perf_counter()
print(f"Run time {toc - tic:0.4f} seconds")

# Printing the performance measures
print('MAE (NUTS) on training data:', metrics.mean_absolute_error(y_train, y_train_pred))
print('MSE (NUTS) on training data:', metrics.mean_squared_error(y_train, y_train_pred))
print('MAE (NUTS) on test data:', metrics.mean_absolute_error(y_test, y_test_pred))
print('MSE (NUTS) on test data:', metrics.mean_squared_error(y_test, y_test_pred))


# # ----------------------------- UNDER CONSTRUCTION ;-) ---------------------------
### DO NOT DELELTE THIS YET
# sns.distplot(y_train - pred1)
# Include a title
# plt.title("Residuals PDF", size=18)
# plt.show()
# # plt.scatter(y_train, pred1)
# # Let's also name the axes
# plt.xlabel('Targets (y_train)',size=18)
# plt.ylabel('Predictions (y_hat)',size=18)
# # Sometimes the plot will have different scales of the x-axis and the y-axis
# # This is an issue as we won't be able to interpret the '45-degree line'
# # We want the x-axis and the y-axis to be the same
# # plt.xlim(6,13)
# # plt.ylim(6,13)
# plt.show()
# # OLS
# from sklearn.linear_model import LinearRegression

# reg = LinearRegression().fit(X_train, y_train)

# pred3=reg.predict(X_train)
# print('MSE (OLS):', metrics.mean_squared_error(y_train, pred3))
