import numpy as np
from cvxopt import matrix, solvers
from sklearn.datasets import load_breast_cancer


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-np.linalg.norm(x-y)**2 / (2 * (sigma ** 2)))

def rbf_kernel(a, b, sigma=0.1):
    return np.exp(-np.square(np.linalg.norm(a - b))/(2 * np.power(sigma,2)))


class SVM:
    def __init__(self, kernel, C=None):
        self.alpha = None
        self.kernel = kernel
        self.C = C
        self.support_threshold = 1e-5

    def calculate_kernel_matrix(self, X):
        num_samples = X.shape[0]
        kernel = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                kernel[i][j] = self.kernel(X[i], X[j])

        return kernel

    def fit(self, X, Y):
        num_samples = X.shape[0]
        num_features = X.shape[1]

        kernel = self.calculate_kernel_matrix(X)

        P = matrix(np.outer(Y, Y) * kernel)
        q = matrix(- np.ones(num_samples))
        A = matrix(Y, (1, num_samples))
        b = matrix(0.0)

        if self.C is None:
            G = matrix(- np.eye(num_samples))
            h = matrix(np.zeros(num_samples))
        else:
            tmp1 = np.diag(np.ones(num_samples) * -1)
            tmp2 = np.identity(num_samples)
            G = matrix(np.vstack((tmp1, tmp2)))
            tmp1 = np.zeros(num_samples)
            tmp2 = np.ones(num_samples) * self.C
            h = matrix(np.hstack((tmp1, tmp2)))

        sol = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(sol['x'])


        mask = self.alpha > self.support_threshold
        index = np.arange(len(self.alpha))[mask]
        self.alpha = self.alpha[mask]
        self.X_support = X[mask]
        self.Y_support = Y[mask]

        print(self.alpha)

        print("{} support vectors out of {} points".format(len(self.alpha), num_samples))

        # Bias parameter
        self.bias = 0
        for i in range(len(self.alpha)):
            self.bias += self.Y_support[i]
            self.bias -= np.sum(self.alpha * self.Y_support * kernel[index[i], mask])
        self.bias /= len(self.alpha)

        if self.kernel == linear_kernel:
            self.w = np.zeros(num_features)
            for n in range(len(self.alpha)):
                self.w += self.alpha[n] * self.Y_support[n] * self.X_support[n]
      
    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            for a, sv_y, sv_x in zip(self.alpha, self.Y_support, self.X_support):
                y_predict[i] += a * sv_y * self.kernel(X[i], sv_x)
        return y_predict + self.bias

    def predict(self, X):
        return np.sign(self.project(X))
