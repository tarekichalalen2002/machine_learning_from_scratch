import numpy as np

def segmond(x):
    return 1/(1+np.exp(-x))

class LogisticRegression:
    def __init__(self, lr = 0.001, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_pred = segmond(np.dot(X, self.weights) + self.bias)
            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred-y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    def predict(self,X):
        y_pred = segmond(np.dot(X, self.weights) + self.bias)
        y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
        return y_pred_cls