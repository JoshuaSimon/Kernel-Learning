import numpy as np
import warnings

class KernelSVM:
    """ 
    Kernel SVM Classifier.
    
    Parameters
    ----------
    kernel : string
        Name the kernel function to be used in the model.
        
    C : float (must be positive)
        Regularization parameter for false classifications.
        
    max_iteration : integer
        Maximum number of solver iterations.
        
    degree : integer
        Degree of the polynomial for polynomial kernels.
        
    gamma : float (must be positive)
        Precision parameters for RBF kernels.
    ----------
    """
    
    def __init__(self, kernel = 'linear', C = 5000.0, max_iteration = 5000, degree = 3, gamma = 1):
        self.kernel = {'poly' : lambda x1,x2: np.dot(x1, x2.T) ** degree,
                       'linear' : lambda x1,x2: np.dot(x1, x2.T),
                       'rbf' : lambda x1,x2: np.exp(-gamma*np.sum((x2-x1[:,np.newaxis])**2,axis=-1))}[kernel]
        self.C = C
        self.max_iteration = max_iteration
        
    # We want to limit our parameter t, so that our 
    # alpha-parameter does not leave the (square) borders.
        
    def restrict_to_square(self, t, v0, u):
        t = (np.clip(v0 + t*u, 0, self.C) - v0)[1]/u[1]
        return (np.clip(v0 + t*u, 0, self.C) - v0)[0]/u[0]
        
    def fit(self, X, y):
        """ 
        Fitting given data by solving the dual optimization problem 
        to determine the weigths and support vectors. A sequential 
        quadratic programming algorithm is used for the optimization.
        
        Parameters
        ----------
        X : matrix_like
            Matrix with (x1, x2) data points.
            
        y : array_like
            Vector with color labels for individual points.
        ----------
        """
        epsilon = 1e-15
        self.x = X.copy()
        
        # Transformation of the classes 0,1 into -1,+1.
        self.y = y * 2 - 1
        self.alpha = np.zeros_like(self.y, dtype=float)
        
        # Calculation of Q matrix (Gram matrix multiplied by y).
        self.Q = self.kernel(self.x, self.x) * self.y[:,np.newaxis] * self.y
            
        # Sequential quadratic programming.
        for _ in range(self.max_iteration):
            # for all alphas
            for idxM in range(len(self.alpha)):
                # choose random idxL
                idxL = np.random.randint(0, len(self.alpha))
                # count formula (4c)
                P = self.Q[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                # count formula (4a)
                v0 = self.alpha[[idxM, idxL]]
                # count formula (4b)
                q0 = 1 - np.sum(self.alpha * self.Q[[idxM, idxL]], axis=1)
                # count formula (4d)
                u = np.array([-self.y[idxL], self.y[idxM]])
                # formula (5) if idxM = idxL
                t_max = np.dot(q0, u) /(np.dot(np.dot(P, u), u) + epsilon)
                self.alpha[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)
                    
                # Determine the support vectors. 
                # Epsilon is used as a numerical threshold since alpha can't be equal to zero.
                idx, = np.nonzero(self.alpha > epsilon)
                
                # Calculate bias value.
                self.b = np.mean((1.0-np.sum(self.Q[idx]*self.alpha,axis=1))*self.y[idx])
                    
    def decision_function(self, x):
        """ Predict y values in {-1, 1}. """
        return np.sum(self.kernel(x, self.x) * self.y * self.alpha, axis=1) + self.b
        
    def predict(self, x):
        """ Transformation of the classes -1,+1 into 0,1. """ 
        return (np.sign(self.decision_function(x)) + 1) // 2





from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as snc; snc.set()
from sklearn.datasets import make_blobs, make_circles
from matplotlib.colors import ListedColormap



def test_plot(X, y, svm_model, axes, title):
    plt.axes(axes)
    xlim = [np.min(X[:, 0]), np.max(X[:, 0])]
    ylim = [np.min(X[:, 1]), np.max(X[:, 1])]
    xx, yy = np.meshgrid(np.linspace(*xlim, num=700), np.linspace(*ylim, num=700))
    rgb = np.array([[210, 0, 0], [0, 0, 150]])/255.0
    
    svm_model.fit(X, y)
    z_model = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='prism') # standard
    plt.contour(xx, yy, z_model, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.contourf(xx, yy, np.sign(z_model.reshape(xx.shape)), alpha=0.3, levels=2, cmap=ListedColormap(rgb), zorder=1)
    plt.title(title)
    

X, y = make_circles(100, factor=.1, noise=.1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    svm_model = KernelSVM(kernel='rbf', C=10, max_iteration=120, gamma=1)
    svm_model.fit(X, y)

"""
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
test_plot(X, y, SVM(kernel='rbf', C=10, max_iteration=120, gamma=1), axs[0], 'SVM Algorithm')
test_plot(X, y, SVC(kernel='rbf', C=10, gamma=1), axs[1], 'sklearn')# 

X,y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=1.4)
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
test_plot(X, y, SVM(kernel='linear', C=10, max_iteration=120), axs[0], 'SVM Algorithm')
test_plot(X, y, SVC(kernel='linear', C=10), axs[1], 'sklearn')#


fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
test_plot(X, y, SVM(kernel='poly', C=10, max_iteration=120, degree=3), axs[0], 'SVM Algorithm')
test_plot(X, y, SVC(kernel='poly', C=10, degree=3), axs[1], 'sklearn')#

plt.show()
"""