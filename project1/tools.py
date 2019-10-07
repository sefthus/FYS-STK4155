import numpy as np
from random import random, seed
import sys


def create_design_matrix(x, y, d=5):
    """ create the design matrix of x and y with a polynomial of degree d """

    x = x.ravel()
    y = y.ravel()

    N = len(x)
    l = int((d+1)*(d+2)/2)
    X = np.ones((N,l))

    for i in range(1,d+1):
        q = int(i*(i+1)/2)

        for n in range(i+1):
            X[:,q+n] = x**(i-n) * y**n
    
    return X

def invert_matrix(X):
    """ inverst matrix X through singular value decomposition"""
    
    U, s, Vt = np.linalg.svd(X)
    Sigma = np.zeros((len(U),len(Vt)))

    for i in range(0,len(Vt)):
        Sigma[i,i] = s[i]
    
    Ut = U.T
    V = Vt.T

    # use pinv, not inv, which has rounding errors
    Sigmainv = np.linalg.pinv(Sigma)
    Xinv = np.matmul(V, np.matmul(Sigmainv,Ut))

    # test if same result = True # REMEMBER TO CHANGE BACK
    #Xinv = np.linalg.pinv(X)

    return Xinv


def MSE_func(ytrue, ypredict):
    """ calculates the mean square error """

    return np.mean((ytrue-ypredict)**2)

def R2_score_func(ytrue, ypredict):
    """ calculates the r2 score """

    return 1 - (np.sum((ytrue-ypredict)**2)/np.sum((ytrue-np.mean(ytrue))**2))
