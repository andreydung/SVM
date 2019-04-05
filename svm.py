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
        self.kernel = kernel
        self.C = C
        self.support_threshold = 1e-5

        self.alpha = None
        self.w = None
        self.bias = None

    def kernel_matrix(self, X):
        num_samples = X.shape[0]
        kernel = np.zeros((num_samples, num_samples))
        for i in range(num_samples):
            for j in range(num_samples):
                kernel[i][j] = self.kernel(X[i], X[j])

        return kernel

    def fit(self, X, Y):
        num_samples = X.shape[0]
        num_features = X.shape[1]

        kernel = self.kernel_matrix(X)

        # solve using CVXOPT optimizer
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

        self.alpha_to_model()

    def alpha_to_model(self):
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


class SMO(SVM):

    def __init__(self):
        super().__init__()


    @staticmethod
    def objective_function(alphas, target, kernel, X_train):
        """Returns the SVM objective function based in the input model defined by:
        `alphas`: vector of Lagrange multipliers
        `target`: vector of class labels (-1 or 1) for training data
        `kernel`: kernel function
        `X_train`: training data for model."""
        return np.sum(alphas) - 0.5 * np.sum((target[:, None] * target[None, :]) * kernel(X_train, X_train) * (alphas[:, None] * alphas[None, :]))


    def fit(self, X, Y):

        num_samples = X.shape[0]
        self.alpha = np.zeros(num_samples)
        self.error = np.zeros_like(self.alpha)
        self.X = X
        self.Y = Y

        numChanged = 0
        examineAll = True

        while(numChanged > 0) or (examineAll):
            numChanged = 0
            if examineAll:
                # loop over all training examples
                for i in range(num_samples):
                    examine_result, model = examine_example(i, model)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = SMO.objective_function(model.alphas, model.y, model.kernel, model.X)
                        model._obj.append(obj_result)
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((model.alphas != 0) & (model.alphas != model.C))[0]:
                    examine_result, model = examine_example(i, model)
                    numChanged += examine_result
                    if examine_result:
                        obj_result = SMO.objective_function(model.alphas, model.y, model.kernel, model.X)
                        model._obj.append(obj_result)
            examineAll = not examineAll
            
        return model

    def examine_example(i):
        
        num_samples = self.X.shape[0]
        y = self.Y[i]
        alph = self.alpha[i]
        E = self.error[i]
        r = E * y

        # Proceed if error is within specified tolerance (tol)
        if ((r < -tol and alph < self.C) or (r > tol and alph > 0)):
            
            if len(self.alpha[(self.alpha != 0) & (self.alpha != self.C)]) > 1:
                # Use 2nd choice heuristic is choose max difference in error
                if self.error[i] > 0:
                    k = np.argmin(self.error)
                else:
                    k = np.argmax(self.error)
                step_result = take_step(k, i)
                if step_result:
                    return True
                
            # Loop through non-zero and non-C alphas, starting at a random point
            for k in np.roll(np.where((self.alpha != 0) & (self.alpha != self.C))[0],
                              np.random.choice(np.arange(num_samples))):
                step_result = take_step(k, i)
                if step_result:
                    return True
            
            # loop through all alphas, starting at a random point
            for k in np.roll(np.arange(num_samples), np.random.choice(np.arange(num_samples))):
                step_result = take_step(k, i)
                if step_result:
                    return True
        
        return False

    def take_step(self, i1, i2):
    
        # Skip if chosen alphas are the same
        if i1 == i2:
            return False
        
        alph1 = self.alpha[i1]
        alph2 = self.alpha[i2]
        y1 = self.Y[i1]
        y2 = self.Y[i2]
        E1 = self.error[i1]
        E2 = self.error[i2]
        s = y1 * y2
        
        # Compute L & H, the bounds on new possible alpha values
        if (y1 != y2):
            L = max(0, alph2 - alph1)
            H = min(self.C, model.C + alph2 - alph1)
        else:
            L = max(0, alph1 + alph2 - model.C)
            H = min(self.C, alph1 + alph2)

        if (L == H):
            return False

        # Compute kernel & 2nd derivative eta
        k11 = self.kernel(self.X[k], self.X[k])
        k12 = self.kernel(self.X[k], self.X[i])
        k22 = self.kernel(self.X[i], self.X[i])
        eta = 2 * k12 - k11 - k22
        
        # Compute new alpha 2 (a2) if eta is negative
        if (eta < 0):
            a2 = alph - y2 * (E1 - E2) / eta
            # Clip a2 based on bounds L & H
            if L < a2 < H:
                a2 = a2
            elif (a2 <= L):
                a2 = L
            elif (a2 >= H):
                a2 = H
                
        # If eta is non-negative, move new a2 to bound with greater objective function value
        else:
            alphas_adj = model.alphas.copy()
            alphas_adj[i] = L
            # objective function output with a2 = L
            Lobj = SMO.objective_function(alphas_adj, model.y, model.kernel, model.X) 
            alphas_adj[i] = H
            # objective function output with a2 = H
            Hobj = SMO.objective_function(alphas_adj, model.y, model.kernel, model.X)
            if Lobj > (Hobj + eps):
                a2 = L
            elif Lobj < (Hobj - eps):
                a2 = H
            else:
                a2 = alph
                
        # Push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (model.C - 1e-8):
            a2 = model.C
        
        # If examples can't be optimized within epsilon (eps), skip this pair
        if (np.abs(a2 - alph) < eps * (a2 + alph + eps)):
            return 0, model
        
        # Calculate new alpha 1 (a1)
        a1 = alph1 + s * (alph - a2)
        
        # Update threshold b to reflect newly calculated alphas
        # Calculate both possible thresholds
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph) * k12 + model.b
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph) * k22 + model.b
        
        # Set new threshold based on if a1 or a2 is bound by L and/or H
        if 0 < a1 and a1 < C:
            b_new = b1
        elif 0 < a2 and a2 < C:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        # Update model object with new alphas & threshold
        model.alphas[k] = a1
        model.alphas[i] = a2
        
        # Update error cache
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alph in zip([k, i], [a1, a2]):
            if 0.0 < alph < model.C:
                model.errors[index] = 0.0
        
        # Set non-optimized errors based on equation 12.11 in Platt's book
        non_opt = [n for n in range(model.m) if (n != k and n != i)]
        model.errors[non_opt] = model.errors[non_opt] + \
                                y1*(a1 - alph1)*model.kernel(model.X[k], model.X[non_opt]) + \
                                y2*(a2 - alph)*model.kernel(model.X[i], model.X[non_opt]) + model.b - b_new
        
        # Update model threshold
        model.b = b_new
        
        return 1, model
