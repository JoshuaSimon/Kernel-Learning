import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

#generating data:
X, y = make_blobs(n_samples=20, centers=2, cluster_std = 4.5 ,random_state=123)

#scatterplot (data)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


# C : float, default=1.0 
# (see https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/svm/_classes.py)
# Regularization parameter. The strength of the regularization is
# inversely proportional to C. Must be strictly positive.
C = [0.00001, 0.0001, 0.01, 1, 10, 1e5, 1e6]

# Creating/fitting model for different C's:
for c in C:
    model = svm.SVC(kernel='linear', C=c)
    model.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y)

    #adding decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    #decision boundary/margin
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
            linestyles=['--', '-', '--'])

    #support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')

    # Print some info to console.
    print(f"Number of support vectors: {len(model.support_vectors_[:, 0])}, C = {c}")

    plt.title(f"C = {c}")
    plt.show()