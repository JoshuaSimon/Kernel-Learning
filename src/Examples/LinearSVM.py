import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

#generating data:
X, y = make_blobs(n_samples=20, centers=2, center_box=(-5.0, 5.0) ,random_state=1)

#creating/fitting model:
model = svm.SVC(kernel='linear')
model.fit(X, y)

# Scatterplot (data).
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()

# Scatterplot with possible hyperplanes.
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.axline((-6.5, 1.0), slope=-0.4, linestyle='--', color='black', label='Hyperplane 1')
plt.axline((-6.5, 4.0), slope=-1.4, linestyle='-.', color='blue', label='Hyperplane 2')
plt.axline((-4.2, 4.0), slope=-1.2, linestyle=':', color='green', label='Hyperplane 3')
plt.legend()
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
Z = model.decision_function(xy).reshape(XX.shape)

#decision boundary/margin
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])

#support vectors
ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='k')

plt.show()