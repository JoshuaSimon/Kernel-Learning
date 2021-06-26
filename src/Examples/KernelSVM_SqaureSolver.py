import numpy as np

class SVM:
    #constructor
    def __init__(self, kernel = 'linear', C = 5000.0, max_iteration = 5000, degree = 3, gamma = 1):
        self.kernel = {'poly' : lambda x,y: np.dot(x, y.T) ** degree,
                       'linear' : lambda x,y: np.dot(x, y.T),
                       'rbf' : lambda x,y: np.exp(-gamma*np.sum((y-x[:,np.newaxis])**2,axis=-1))}[kernel]
        self.C = C
        self.max_iteration = max_iteration
        
    #we want to limit our parameter t, that our lamda-parameter doesn't leaving the (square) borders
        
    def restrict_to_square(self, t, v0, u):
        t = (np.clip(v0 + t*u, 0, self.C) - v0)[1]/u[1]
        return (np.clip(v0 + t*u, 0, self.C) - v0)[0]/u[0]
        
    def fit(self, x, y):
        self.x = x.copy()
        #transformation of the classes 0,1 into -1,+1
        self.y = y * 2 - 1
        self.alpha = np.zeros_like(self.y, dtype=float)
        
        #formula (3). Count the matrix K
        self.K = self.kernel(self.x, self.x) * self.y[:,np.newaxis] * self.y
            
        #self.max_iteration
        for _ in range(self.max_iteration):
            #for all aplhas
            for idxM in range(len(self.alpha)):
                # choose random idxL
                idxL = np.random.randint(0, len(self.alpha))
                # count formula (4c)
                Q = self.K[[[idxM, idxM], [idxL, idxL]], [[idxM, idxL], [idxM, idxL]]]
                # count formula (4a)
                v0 = self.alpha[[idxM, idxL]]
                # count formula (4b)
                k0 = 1 - np.sum(self.alpha * self.K[[idxM, idxL]], axis=1)
                # count formula (4d)
                u = np.array([-self.y[idxL], self.y[idxM]])
                # formula (5) if idxM = idxL
                t_max = np.dot(k0, u) /(np.dot(np.dot(Q, u), u) + 1E-15)
                self.alpha[[idxM, idxL]] = v0 + u * self.restrict_to_square(t_max, v0, u)
                    
                #support vektor index
                idx, = np.nonzero(self.alpha > 1E-15)
                #count formula (1)
                self.b = np.mean((1.0-np.sum(self.K[idx]*self.alpha,axis=1))*self.y[idx])
                    
    def decision_function(self, x):
        return np.sum(self.kernel(x, self.x) * self.y * self.alpha, axis=1) + self.b
        
    def predict(self, x):
            #transformation of the classes -1,+1 into 0,1
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
svm_model = SVM(kernel='rbf', C=10, max_iteration=120, gamma=1)
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