from sklearn.datasets import make_blobs, make_moons, make_circles
import matplotlib.pyplot as plt
import pandas as pd


#### LINEAR SEPERABLE DATA

#X, y = make_blobs(n_samples=20, centers=2, center_box=(-5.0, 5.0) ,random_state=1)

#plt.scatter(X[:, 0], X[:, 1], c=y)
#plt.show()

#linearshape = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
#linearshape.to_csv('linearshape.csv')


#### MOON SHAPE ####


#X, y = make_moons(n_samples = 100, noise = 0.1, random_state = 123)

#plt.scatter(X[:, 0], X[:, 1], c = y)
#plt.show()

#moonshape = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
#moonshape.to_csv('moonshape.csv')


#### CIRCLE SHAPE ####


#X, y = make_circles(n_samples = 100, noise = 0.05, random_state = 321)

#plt.scatter(X[:, 0], X[:, 1], c = y)
#plt.show()

#circleshape = pd.DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
#circleshape.to_csv('circleshape.csv')