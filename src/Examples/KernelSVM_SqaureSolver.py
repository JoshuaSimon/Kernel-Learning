import numpy as np
import warnings

from time import time

from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits import mplot3d


class KernelSVM:
    """ 
    Kernel SVM Classifier.
    
    Parameters
    ----------
    kernel : string
        Name of the kernel function to be used in the model.
        
    C : float (must be positive)
        Regularization parameter for false classifications.
        
    max_iteration : integer
        Maximum number of solver iterations.
        
    random_state : boolean
        If true, a seed is used in the optimization. 
        
    degree : integer
        Degree of the polynomial for polynomial kernels.
        
    coeff : float (must be positive)
        Scale parameter for polynomial kernels.
        
    gamma : float (must be positive)
        Precision parameter for RBF kernels.
    ----------
    """
    
    def __init__(self, kernel = 'linear', C=5000.0, max_iteration=100, random_state=False, 
                 degree=3, coeff=0, gamma=1):
        self.kernel = {
            'linear' : lambda x1,x2: np.dot(x1, x2.T),
            'poly' : lambda x1,x2: (np.dot(x1, x2.T) + coeff) ** degree,
            'rbf' : lambda x1,x2: np.exp(-gamma*np.sum((x2-x1[:,np.newaxis])**2,axis=-1))}[kernel]
        self.C = C
        self.max_iteration = max_iteration
        self.random_state = random_state
        
    def fit(self, X, y):
        """ 
        Fitting given data by solving the dual optimization problem 
        to determine the weigths and support vectors. A sequential 
        quadratic programming algorithm is used for the optimization.
        
        Parameters
        ----------
        X : matrix_like
            Matrix with (x1, ... , xn) data points.
            
        y : array_like
            Vector with labels for individual points.
        ----------
        """
        
        def restrict_to_square(t, v0, u):
            t = (np.clip(v0 + t*u, 0, self.C) - v0)[1]/u[1]
            return (np.clip(v0 + t*u, 0, self.C) - v0)[0]/u[0]
        
        epsilon = 1e-15
        seed = 0
        self.x = X.copy()
        self.solution_distances = []
        
        # Transformation of the classes {0, 1} into {-1, 1}.
        self.y = y * 2 - 1
        self.alpha = np.zeros_like(self.y, dtype=float)
        self.convergence_step = np.zeros_like(self.y, dtype=float)
        
        # Calculation of Q matrix (Gram matrix multiplied by y).
        self.Q = self.kernel(self.x, self.x) * self.y[:,np.newaxis] * self.y
            
        # Sequential minimal optimization.
        for _ in range(self.max_iteration):
            self.convergence_step = self.alpha
            seed += len(self.alpha)
            
            # Iterate over all alphas.
            for idxM in range(len(self.alpha)):
                # choose random idxL
                if self.random_state:
                    np.random.seed(seed)
                idxL = np.random.randint(0, len(self.alpha))
                # Calculate formula (23)
                P = self.Q[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                # Calculate formula (21)
                v0 = self.alpha[[idxM, idxL]]
                # Calculate formula (22)
                q0 = 1 - np.sum(self.alpha * self.Q[[idxM, idxL]], axis=1)
                # Calculate formula (24)
                u = np.array([-self.y[idxL], self.y[idxM]])
                # Calculate formula (25)
                t_max = np.dot(q0, u) / (np.dot(np.dot(P, u), u) + epsilon)
                self.alpha[[idxM, idxL]] = v0 + u * restrict_to_square(t_max, v0, u)
                seed += 1
                
                #self.solution_distances.append(np.linalg.norm(self.convergence_step - self.alpha, ord=2))
                self.solution_distances.append(self.alpha)
                    
        # Determine the support vectors. 
        # Epsilon is used as a numerical threshold since alpha can't be equal to zero.
        support_indices = self.alpha > epsilon
        self.support_vectors_ = X[support_indices]
        
        # Calculate bias value.
        self.b = np.mean((1.0-np.sum(self.Q[support_indices]*self.alpha,axis=1))*self.y[support_indices])
                    
    def decision_function(self, x):
        """ Predict y values in {-1, 1}. """
        return np.sum(self.kernel(x, self.x) * self.y * self.alpha, axis=1) + self.b
        
    def predict(self, x):
        """ Transformation of the prediction classes {-1, 1} into {0, 1}. """ 
        return (np.sign(self.decision_function(x)) + 1) // 2


def plot_2D(X, y, legend=False):
    """ 
    Standard 2D scatter plot. 
    -----------
    Parameters:
    - X: matrix with (x1, x2) data points
    - y: vector with color labels for individual points
    - legend(default = False): if true, legend for labels is shown
    -----------
    """
    # Map labels to -1 and + 1.
    y = np.where(y == 0, -1, y)
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y)
    
    if legend:
        plt.legend(*scatter.legend_elements(),
                    loc="upper left", title="Labels")
    

def plot_2D_svm(model):
    """ 
    Draws the decision boundary/margin of a svm model by 
    estimating more data points of the grid and marks the 
    support vectors of the passed svm model.
    -----------
    Parameters:
    - model: sklearn.svm.SVC() model object
    -----------
    """
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Estimate decision boundary.
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # Plot decision boundary/margin.
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # Plot support vectors.
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k', 
               label="Support Vectors")


def plot_3D(X, y, z, elev=30, azim=30):
    """ 
    Plots rotatable 3D scatter plot with scaled z-axis. 
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
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_zlabel(r"$z$")


def compare_svm_models(X, y, names, models):
    i = 0
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
    for model in models:
        training_time = time()
        model.fit(X=X, y=y)
        training_time = time() - training_time

        print(f"{names[i]}: Elapsed time for model training: {training_time} seconds.")
        print(f"{names[i]}: Number of support vectors: {len(model.support_vectors_[:, 0])} \n")
        
        plt.subplot(121 + i)
        plot_2D(X, y)
        plot_2D_svm(model)
        plt.title(names[i])
        
        i += 1


def calculate_scores(model, X_test, y_test):
    """ 
    Calculates the confusion matrix for a model to determine
    accuracy, sensitivity and specificity scores by true 
    positve (tp), true negative (tn), false positive (fp) and
    false negative (fn) classifications.
    
    Parameters
    ----------
    model : model object
        A binary classification model with a predict function.
    
    X_test : matrix_like
        Matrix with (x1, ... , xn) data points.

    y_test : array_like
        Vector with labels for individual points.
    ----------
    """
    
    y_predict = model.predict(X_test)
    conf_mat = metrics.confusion_matrix(y_test, y_predict)
    tp, tn, fp, fn = conf_mat[0,0], conf_mat[1,1], conf_mat[1,0], conf_mat[0,1]
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (fp + tn)
    return (accuracy, sensitivity, specificity)


def compare_models(X, y, names, classifiers):
    """ 
    Compares a list of classification models by 
    different scores and visual results.
    
    Parameters
    ----------
    X : matrix_like
        Matrix with (x1, ... , xn) data points.

    y : array_like
        Vector with labels for individual points.
        
    names : list 
        A list with strings of model names.
        
    classifiers : list
        A list of binary classification models.
    ----------
    """
    
    # Split data in train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)
    
    # Plotting and layout.
    i = 0
    figure = plt.figure(figsize=(27, 9))
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(141 + i)
    ax.set_title("Input data")
    
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    
    ax.set_xticks(())
    ax.set_yticks(())
    
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    
    i += 1

    # Fit each model in the list, calculate scores and plot the results.
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(141 + i)
        clf.fit(X_train, y_train)
        
        # Calculate confusion matrix and scores.
        accuracy, sensitivity, specificity =  calculate_scores(clf, X_test, y_test)
        
        print(f"Model: {name}")
        print(f"Scores: accuracy: {accuracy}, sensitivity: {sensitivity}, specificity: {specificity} \n")

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max] x [y_min, y_max].
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(XX.shape)
        ax.contourf(XX, YY, Z, cmap=cm, alpha=.8)

        # Plot the training points.
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        
        # Plot the testing points.
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(name)

        i += 1


if __name__ == "__main__":
    # Generate data.
    X_moon, y_moon = make_moons(n_samples = 200, noise = 0.1, random_state = 123)
    X_circle, y_circle = make_circles(n_samples = 200, factor = 0.4, noise = 0.1, random_state = 321)

    # Initialize the model.
    sk_learn = svm.SVC(kernel='rbf', gamma=2, C=1)
    kernel = KernelSVM(kernel="rbf", C=1, max_iteration=100, gamma=2, random_state=True)

    # Compare models.
    names = ["Scikit Learn SVM", "Kernel SVM"]
    models = [sk_learn, kernel]

    compare_svm_models(X_moon, y_moon, names, models)
    #compare_models(X_moon, y_moon, names, models)
    #compare_svm_models(X_circle, y_circle, names, models)
    #plt.show()

    dist = []
    for sol in kernel.solution_distances:
        dist.append(np.linalg.norm(sol, ord=2))
        #print(sol)

    #plt.plot(dist)
    #plt.show()
