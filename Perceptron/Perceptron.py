import numpy as np

def unit_step_func(x):
    return np.where(x>=0, 1, 0)


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.activation_func = unit_step_func
        self.weights = None
        self.bias = None
    
    def fit(self,X,y):
        n_samples, n_features = X.shape

        self.weights = np.random.rand(n_features)
        self.bias = 0

        y_ = np.where(y>0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_func(linear_output)
                perceptron_update = self.lr * (y_[idx] - y_predicted) 
                self.weights += perceptron_update * x_i
                self.bias += perceptron_update

    def predict(self,X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_func(linear_output)
        return y_predicted