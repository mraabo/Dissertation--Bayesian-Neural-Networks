import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from HMC_NN_class import HMC_neural_network

load = pd.read_csv('imdb_5000_movies.csv')

# Select only movies produced in USA
load = load[load['production_countries'].str.contains(
    "United States of America")]

# Remove examples with 0 budget, revenue or vote_average
load = load[['budget', 'revenue', 'vote_average']]
load = load.replace(0, pd.np.nan).dropna(axis=0, how='any')

# budget and revenue in millionsty
x = np.array([load['budget'], load['revenue']])/10**6
y = np.array(load['vote_average'])

# --------------------------------- Visualize data ------------------------------------
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(x[0], x[1], y,
#              c=y, cmap='copper', marker='o')
# plt.show()

# --------------------------------- Parameters ------------------------------------
num_hidden_neurons = 16
weight_prior_var = 1
weight_prior_mean = 0
weight_likelihood_stddev = 16

# --------------------------------- Instantiate and sample from class ------------------------------------

My_HMC_NN = HMC_neural_network(x, y, num_hidden_neurons,
                               weight_prior_mean, weight_prior_var, weight_likelihood_stddev)

initial_state = np.ones(num_hidden_neurons * (x.shape[0] + 2)+1)
My_HMC_NN.create_HMC_kernel()
samples, kernel_results = My_HMC_NN.run_chain(initial_state)
print("Acceptance rate:", kernel_results.is_accepted.numpy().mean())

# --------------------------------- Predict and visualize results ------------------------------------

y_pred = My_HMC_NN.predict(samples)
y_pred_mean = y_pred.mean(axis=0)
print("MSE:", np.sum((y_pred_mean - y)**2))

# Visualize the data set
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x[0], x[1], y,
             c=y, cmap='copper', marker='o')
ax.scatter3D(x[0], x[1], y_pred_mean,
             c=y_pred_mean, cmap='copper', marker='x')
plt.show()
