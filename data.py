import numpy as np
from sklearn.model_selection import train_test_split


def gen_lin_separable_data(mean1=np.array([0, 2]), 
                           mean2=np.array([2, 0]), 
                           cov=np.array([[0.8, 0.6], [0.6, 0.8]]),
                           num=200):
    # generate training data in the 2-d case
    X1 = np.random.multivariate_normal(mean1, cov, num)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, num)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def gen_non_lin_separable_data(mean1=np.array([-1, 2]),
                               mean2=np.array([1, -1]),
                               mean3=np.array([4, -4]),
                               mean4=np.array([-4, 4]),
                               cov=np.array([[1.0,0.8], [0.8, 1.0]])):
    X1 = np.random.multivariate_normal(mean1, cov, 50)
    X1 = np.vstack((X1, np.random.multivariate_normal(mean3, cov, 50)))
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 50)
    X2 = np.vstack((X2, np.random.multivariate_normal(mean4, cov, 50)))
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def gen_lin_separable_overlap_data():
    # generate training data in the 2-d case
    mean1 = np.array([0, 2])
    mean2 = np.array([2, 0])
    cov = np.array([[1.5, 1.0], [1.0, 1.5]])
    X1 = np.random.multivariate_normal(mean1, cov, 100)
    y1 = np.ones(len(X1))
    X2 = np.random.multivariate_normal(mean2, cov, 100)
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2


def split(X1, y1, X2, y2, test_size=0.1):
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=test_size)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=test_size)

    X_train = np.vstack((X1_train, X2_train))
    y_train = np.hstack((y1_train, y2_train))

    X_test = np.vstack((X1_test, X2_test))
    y_test = np.hstack((y1_test, y2_test))

    return X_train, y_train, X_test, y_test
