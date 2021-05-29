from sklearn.datasets import make_blobs, make_moons, make_circles
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def rbf(X):
    """ Radial bais function for scaling data in 3rd dimension. """
    return np.exp(-(X ** 2).sum(1))


def grbf(x1, x2, gamma='scale'):
    """ Gaussian radial basis function for RBF kernels. """
    if gamma == 'auto':
        gamma = 1 / len(x1)
    elif gamma == 'scale':
        gamma = 1 / (len(x1) * np.var(x1))

    diff = x1 - x2
    return np.exp(-1.0 * gamma * np.dot(diff, diff) / 2)


def gramian(X, kernel):
    """ Calculates the Gram matrix for a given kernel. """
    return np.apply_along_axis(lambda x1: np.apply_along_axis(lambda x2: kernel(x1, x2), 1, X), 1, X)


def plot_2D(X, y):
    """ Standard 2D scatter plot. """
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


def plot_3D(X, y, z, elev=30, azim=30):
    """ Plots rotatable 3D scatter plot with scaled z-axis. 
        -----------
        Parameters:
        - X: matrix with (x1, x2) data points
        - y: vector with color labels for individual points
        - z: vector with scale levels for individual points
        - elev, azim: view angles
        -----------
    """
    ax = plt.subplot(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], z, c=y, s=50)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('z')
    plt.show()


#### LINEAR SEPERABLE DATA

X, y = make_blobs(n_samples=20, centers=2, center_box=(-5.0, 5.0) ,random_state=1)
#plot_2D(X, y)

#linearshape = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
#linearshape.to_csv('linearshape.csv')


#### MOON SHAPE ####

X, y = make_moons(n_samples = 100, noise = 0.1, random_state = 123)
#plot_2D(X, y)
#plot_3D(X=X, y=y, z=rbf(X))
#plot_3D(X=X, y=y, z=gramian(X, grbf).sum(1))

#moonshape = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
#moonshape.to_csv('moonshape.csv')


#### CIRCLE SHAPE ####

X, y = make_circles(n_samples = 100, factor = 0.4, noise = 0.1, random_state = 321)
plot_2D(X, y)
plot_3D(X=X, y=y, z=rbf(X))
plot_3D(X=X, y=y, z=gramian(X, grbf).sum(1))

#circleshape = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
#circleshape.to_csv('circleshape.csv')


