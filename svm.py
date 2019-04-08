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
    def __init__(self, kernel_function, C):
        self.kernel_function = kernel_function
        self.C = C

        self.alpha = None
        self.w = None
        self.bias = None

        self.X = None
        self.Y = None

    def kernel(self, X1, X2):
        if X1.ndim == 1 and X2.ndim == 1:
            return self.kernel_function(X1, X2)
        elif X1.ndim == 1:
            K = np.zeros(len(X2))
            for i in range(len(X2)):
                K[i] = self.kernel_function(X1, X2[i])
            return K
        elif X2.ndim == 1:
            K = np.zeros(len(X1))
            for i in range(len(X1)):
                K[i] = self.kernel_function(X1[i], X2)
            return K
        else:
            K = np.zeros((X1.shape[0], X2.shape[0]))
            for i in range(X1.shape[0]):
                for j in range(X2.shape[0]):
                    K[i][j] = self.kernel_function(X1[i], X2[j])
            return K

    def objective(self, alpha, X, Y):
        """Returns the SVM objective function based in the input model defined by:"""
        return np.sum(alpha) - 0.5 * np.sum(np.outer(Y, Y) * self.kernel(X, X) * np.outer(alpha, alpha))

    def project(self, X_train, Y_train, X_test):
        return np.dot(self.alpha * Y_train, self.kernel(X_train, X_test)) - self.bias

    def predict(self, X_test):
        return np.sign(self.project(self.X, self.Y, X))

    def keep_support_vector(self, support_threshold=1e-5):
        mask = self.alpha > support_threshold
        self.alpha = self.alpha[mask]
        self.X = self.X[mask]
        self.Y = self.Y[mask]


class SMO(SVM):

    def __init__(self, kernel_function, C=1000):
        super().__init__(kernel_function, C)
        self.tol = 0.01 # error tolerance
        self.eps = 0.01 # alpha tolerance

        # error cache
        self.error = None

    def fit(self, X, Y):
        num_samples = len(Y)
        self.alpha = np.zeros_like(Y)
        self.bias = 0
        self.X = X
        self.Y = Y

        self.error = self.project(X, Y, X) - self.Y

        numChanged = 0
        examineAll = True

        while (numChanged > 0) or (examineAll):
            numChanged = 0
            if examineAll:
                # loop over all training examples
                for i in range(num_samples):
                    examine_result = self.examine_example(i)
                    numChanged += examine_result
            else:
                # loop over examples where alphas are not already at their limits
                for i in np.where((self.alpha != 0) & (self.alpha != self.C))[0]:
                    examine_result = self.examine_example(i)
                    numChanged += examine_result

            if examine_result:
                obj_result = self.objective(self.alpha, self.X, self.Y)

            if examineAll:
                examineAll = False
            elif numChanged == 0:
                examineAll = True        

    def examine_example(self, i):
        
        num_samples = self.X.shape[0]
        y = self.Y[i]
        alph = self.alpha[i]
        E = self.error[i]
        r = E * y

        # Proceed if error is within specified tolerance (tol)
        if ((r < -self.tol and alph < self.C) or (r > self.tol and alph > 0)):
            
            if len(self.alpha[(self.alpha != 0) & (self.alpha != self.C)]) > 1:
                # Use 2nd choice heuristic is choose max difference in error
                if self.error[i] > 0:
                    k = np.argmin(self.error)
                else:
                    k = np.argmax(self.error)
                step_result = self.take_step(k, i)
                if step_result:
                    return True
                
            # Loop through non-zero and non-C alphas, starting at a random point
            for k in np.roll(np.where((self.alpha != 0) & (self.alpha != self.C))[0],
                              np.random.choice(np.arange(num_samples))):
                step_result = self.take_step(k, i)
                if step_result:
                    return True
            
            # loop through all alphas, starting at a random point
            for k in np.roll(np.arange(num_samples), np.random.choice(np.arange(num_samples))):
                step_result = self.take_step(k, i)
                if step_result:
                    return True
        
        return False

    def take_step(self, i1, i2):

        # print("Pair: {}, {}".format(i1, i2))
    
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
            H = min(self.C, self.C + alph2 - alph1)
        else:
            L = max(0, alph1 + alph2 - self.C)
            H = min(self.C, alph1 + alph2)

        if (L == H):
            return False

        # Compute kernel & 2nd derivative eta
        k11 = self.kernel(self.X[i1], self.X[i1])
        k12 = self.kernel(self.X[i1], self.X[i2])
        k22 = self.kernel(self.X[i2], self.X[i2])
        eta = 2 * k12 - k11 - k22
    
        # Compute new alpha 2 (a2) if eta is negative
        if (eta < 0):
            a2 = alph2 - y2 * (E1 - E2) / eta
            a2 = np.clip(a2, L, H)
                
        # If eta is non-negative, move new a2 to bound with greater objective function value
        else:
            alphas_adj = self.alpha.copy()
            alphas_adj[i] = L
            # objective function output with a2 = L
            Lobj = self.objective(alphas_adj, self.X, self.Y) 
            alphas_adj[i] = H
            # objective function output with a2 = H
            Hobj = self.objective(alphas_adj, self.X, self.Y)

            if Lobj > (Hobj + self.eps):
                a2 = L
            elif Lobj < (Hobj - self.eps):
                a2 = H
            else:
                a2 = alph
                
        # Push a2 to 0 or C if very close
        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C
        
        # If examples can't be optimized within epsilon (eps), skip this pair
        if (np.abs(a2 - alph2) < self.eps * (a2 + alph2 + self.eps)):
            return False
        
        # Calculate new alpha 1 (a1)
        a1 = alph1 + s * (alph2 - a2)
        
        # Update threshold b to reflect newly calculated alphas
        # Calculate both possible thresholds
        b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + self.bias
        b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + self.bias
        
        # Set new threshold based on if a1 or a2 is bound by L and/or H
        if 0 < a1 and a1 < self.C:
            b_new = b1
        elif 0 < a2 and a2 < self.C:
            b_new = b2
        # Average thresholds if both are bound
        else:
            b_new = (b1 + b2) * 0.5

        # Update model object with new alphas & threshold
        self.alpha[i1] = a1
        self.alpha[i2] = a2
        
        # Update error cache
        # Error cache for optimized alphas is set to 0 if they're unbound
        for index, alph in zip([i1, i2], [a1, a2]):
            if 0.0 < alph < self.C:
                self.error[index] = 0.0
        
        # Set non-optimized errors based on equation 12.11 in Platt's book
        non_opt = [n for n in range(len(self.X)) if (n != i1 and n != i2)]
        self.error[non_opt] = self.error[non_opt] + \
                                y1*(a1 - alph1)*self.kernel(self.X[i1], self.X[non_opt]) + \
                                y2*(a2 - alph2)*self.kernel(self.X[i2], self.X[non_opt]) + self.bias - b_new
        
        # Update model threshold
        self.bias = b_new
        
        return True
