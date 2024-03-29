# # ----------------------------- IMPORTS ---------------------------

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

# # ----------------------------- Defining functions ---------------------------
# Defining target probability


def p(x):
    sigma = np.array([[1, 0.6], [0.6, 1]])  # Covariance matrix
    return ss.multivariate_normal.pdf(x, cov=sigma)


# # ----------------------------- Sampling  ---------------------------
samples = np.zeros((1000, 2))
np.random.seed(42)

x = np.array([7, 0])
for i in range(1000):
    samples[i] = x
    # Gaussian proposal for symmetry
    x_prime = np.random.multivariate_normal(
        mean=x, cov=np.eye(2), size=1).flatten()
    acceptance_prob = min(1, (p(x_prime)) / (p(x)))
    u = np.random.uniform(0, 1)
    if u <= acceptance_prob:
        x = x_prime
    else:
        x = x

# # ----------------------------- Vizualising  ---------------------------

# For vizualising normal contours
X, Y = np.mgrid[-3:3:0.05, -3:3:0.05]
X, Y = np.mgrid[-3:3:0.05, -3:3:0.05]
XY = np.empty(X.shape + (2,))
XY[:, :, 0] = X
XY[:, :, 1] = Y
target_distribution = ss.multivariate_normal(
    mean=[0, 0], cov=[[1, 0.6], [0.6, 1]])


plt.subplot(2, 2, 1)  # row 1, col 2 index 1
plt.contour(X, Y, target_distribution.pdf(XY), cmap=plt.cm.Blues)
plt.ylim(-3, 5)
plt.xlim(-3, 8)
plt.subplot(2, 2, 2)  # index 2
plt.plot(samples[0:100, 0], samples[0:100, 1], 'ro-', color="navy",
         linewidth=.2, markersize=.7, label="First 100 samples")
plt.contour(X, Y, target_distribution.pdf(XY), cmap=plt.cm.Blues)
plt.legend(loc="upper right", fontsize=9)
plt.ylim(-3, 5)
plt.xlim(-3, 8)
plt.subplot(2, 2, 3)  # index 3
plt.plot(samples[0:200, 0], samples[0:200, 1], 'ro-', color="navy",
         linewidth=.2, markersize=.7, label="First 200 samples")
plt.contour(X, Y, target_distribution.pdf(XY), cmap=plt.cm.Blues)
plt.legend(loc="upper right", fontsize=9)
plt.ylim(-3, 5)
plt.xlim(-3, 8)
plt.subplot(2, 2, 4)  # index 4
plt.plot(samples[0:300, 0], samples[0:300, 1], 'ro-', color="navy",
         linewidth=.2, markersize=.7, label="First 300 samples")
plt.contour(X, Y, target_distribution.pdf(XY), cmap=plt.cm.Blues)
plt.legend(loc="upper right", fontsize=9)
plt.ylim(-3, 5)
plt.xlim(-3, 8)
plt.savefig("metro_example.pdf")
plt.show()
