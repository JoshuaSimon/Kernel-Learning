import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles

#generating moon-shaped data
X, y = make_circles(n_samples = 100, noise = 0.05, random_state = 321)

#linear SVM
lmodel = svm.SVC(kernel='linear')
lmodel.fit(X, y)

#scatterplot (data)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y)

#adding decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = lmodel.decision_function(xy).reshape(XX.shape)

#decision boundary/margin
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

#support vectors
ax.scatter(lmodel.support_vectors_[:, 0], lmodel.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.show()

#linear SVM
nlmodel = svm.SVC(kernel='rbf')
nlmodel.fit(X, y)

#scatterplot (data)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y)

#adding decision function
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = nlmodel.decision_function(xy).reshape(XX.shape)

#decision boundary/margin
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

#support vectors
ax.scatter(nlmodel.support_vectors_[:, 0], nlmodel.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.show()