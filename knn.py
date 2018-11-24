import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass

    def train(self,  X, y):
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X):
        ntest = X.shape[0]
        Ypred = np.zeros(ntest, dtype = self.ytr.dtype)

        for i in range(ntest):
            dist = np.sum(np.abs(self.Xtr = X[i, :]), axis = 1)
            min_idx = np.argmin(dist)
            Ypred[i] = self.ytr[min_idx]

        return Ypred

