# References:
# -> http://www.pyopt.org/reference/optimizers.slsqp.html
# -> https://en.wikipedia.org/wiki/Sequential_quadratic_programming
# -> http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.512.2567&rep=rep1&type=pdf
# -> https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
# -> https://tonio73.github.io/data-science/classification/ClassificationSVM.html
# -> https://www.csie.ntu.edu.tw/~cjlin/papers/libsvm.pdf
# -> https://scikit-learn.org/stable/modules/svm.html

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from pandas import read_csv
from functools import partial, update_wrapper


class KernelSVM:
    """ 
    Kernel SVM Classifier.
    Takes a margin parameter C and a kernel funtion as
    input arguments.
    """
    def __init__(self, kernel, C=1):
        self.C = C
        self.kernel = kernel
        self.alpha = None
        self.supportVectors = None

    def fit(self, X, y):
        """ 
        Fitting given data by solving the dual optimization problem 
        to determine the weigths and support vectors.
        """
        m = len(y)

        # Calculate the Gram matrix (kernel matrix) multiplied by y.
        hXX = np.apply_along_axis(lambda x1: np.apply_along_axis(
                                    lambda x2: self.kernel(x1,x2), 1, X), 
                                    1, X)
        yp = y.reshape(-1, 1)
        GramHXy = hXX * np.matmul(yp, yp.T)

        # Lagrange dual problem.
        def lagrangianDual(Gram, alpha):
            return alpha.sum() - 0.5 * alpha.dot(alpha.dot(Gram))

        # Partial derivative of the Lagrangian with respect to alpha.
        def lagrangianAlpha(Gram, alpha):
            return np.ones_like(alpha) - alpha.dot(Gram)

        # Constraints on alpha in the shape of:
        # ->  d - C*alpha  = 0
        # ->  b - A*alpha >= 0
        A = np.vstack((-np.eye(m), np.eye(m)))             
        b = np.hstack((np.zeros(m), self.C * np.ones(m)))
        constraints = ({'type': 'eq',   'fun': lambda a: np.dot(a, y),     'jac': lambda a: y},
                       {'type': 'ineq', 'fun': lambda a: b - np.dot(A, a), 'jac': lambda a: -A})

        # Solve the dual optimization problem.
        # To calculate the maximum, we can minimize the opposite.
        # Since our problem is a quadratic programming problem, we can
        # use the "SLSQP" solver method. 
        # SLSQP optimizer is a sequential least squares programming algorithm 
        # which uses the Han–Powell quasi–Newton method with a BFGS update 
        # of the B–matrix and an L1–test function in the step–length algorithm.
        optRes = optimize.minimize(fun=lambda a: -lagrangianDual(GramHXy, a),
                                   x0=np.ones(m), 
                                   method="SLSQP", 
                                   jac=lambda a: -lagrangianAlpha(GramHXy, a), 
                                   constraints=constraints)
        self.alpha = optRes.x
        
        # Determine the support vectors. 
        # Epsilon is used as a numerical threshold since alpha can't be equal to zero.
        epsilon = 1e-8
        supportIndices = self.alpha > epsilon
        self.supportVectors = X[supportIndices]
        self.supportAlphaY = y[supportIndices] * self.alpha[supportIndices]

    def predict(self, X):
        """ Predict y values in {-1, 1}. """
        def predict1(x):
            x1 = np.apply_along_axis(lambda s: self.kernel(s, x), 1, self.supportVectors)
            x2 = x1 * self.supportAlphaY
            return np.sum(x2)
        
        d = np.apply_along_axis(predict1, 1, X)
        return 2 * (d > 0) - 1


def linear(x1, x2):
    """ Canonial dot product for linear kernels. """
    return np.dot(x1, x2)


def dim_plus_1(x1, x2):
    """ Experimental kernel. """
    return np.dot(x1, x2) + np.dot(x1, x1) * np.dot(x2, x2)


def polynomial(x1, x2, d=3, r=0, gamma='auto'):
    """ Polynomial kernel function. """
    if gamma == 'auto':
        gamma = 1 / len(x1)
    elif gamma == 'scale':
        gamma = 1 / (len(x1) * np.var(x1))

    return (gamma * np.dot(x1, x2) + r)**d


def grbf(x1, x2, gamma='scale'):
    """ Gaussian radial basis function for RBF kernels. """
    if gamma == 'auto':
        gamma = 1 / len(x1)
    elif gamma == 'scale':
        gamma = 1 / (len(x1) * np.var(x1))

    diff = x1 - x2
    return np.exp(-1.0 * gamma * np.dot(diff, diff) / 2)


def sigmoid(x1, x2, r=0, gamma='auto'):
    """ Sigmoid kernel function. """
    if gamma == 'auto':
        gamma = 1 / len(x1)
    elif gamma == 'scale':
        gamma = 1 / (len(x1) * np.var(x1))

    return np.tanh(gamma * np.dot(x1, x2) + r)


def plot_SVM(xTrain, yTrain, x0Predict, x1Predict, yPredict, support):
    """ Plots training data and support vectors with decision boundary. """
    plt.scatter(xTrain[:,0], xTrain[:,1], c=yTrain, 
                label="Training Data")
    ax = plt.gca()            
    ax.scatter(support[:,0], support[:,1], s=100, 
                linewidth=1, facecolor='none', edgecolor="k", 
                label="Support Vectors")
    ax.contour(x0Predict, x1Predict, yPredict, 
                colors='k', levels=[-1, 0], alpha=0.3, 
                linestyles=['--', '-']);
    plt.legend()
    plt.show()


def wrapped_partial(func, *args, **kwargs):
    """ Wraps a partial function to set attributes like __name__ and __doc__. """
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func



if __name__ == "__main__":
    # Read training data.
    filenames = ["src/examples/data/circleshape.csv",
                "src/examples/data/moonshape.csv",
                "src/examples/data/linearshape.csv"]
    try:
        data = read_csv(filenames[1])
    except FileNotFoundError: 
        print("Couldn't find file: %s. Please check the file path." % (filenames[1]))
        exit()

    # Map training data.
    xTrain = np.c_[np.array(data["x"]), np.array(data["y"])]
    yTrain = np.array(data["label"])
    yTrain = np.where(yTrain == 0, -1, yTrain)

    # Train the SVM model using different kernel functions.
    kernels = [linear, dim_plus_1,
                wrapped_partial(polynomial, d=3, r=0.0, gamma='auto'), 
                wrapped_partial(grbf, gamma='scale'), 
                wrapped_partial(sigmoid, r=0.0, gamma='auto')]

    for kernel in kernels:
        model = KernelSVM(kernel=kernel, C=10)
        model.fit(xTrain, yTrain)
        support = model.supportVectors

        print("Kernel: %s -> Number of support vectors = %d" % (kernel.__name__, (len(support))))

        # Estimate decision boundary.
        xx = np.linspace(min(xTrain[:,0]), max(xTrain[:,0]), 100)
        yy = np.linspace(min(xTrain[:,1]), max(xTrain[:,1]), 100)
        x0Predict, x1Predict = np.meshgrid(xx, yy)
        xPredict = np.vstack([x0Predict.ravel(), x1Predict.ravel()]).T
        yPredict = model.predict(xPredict).reshape(x0Predict.shape)

        # Plot the results of the fitted model w ith decision boundary.
        plot_SVM(xTrain, yTrain, x0Predict, x1Predict, yPredict, support)
