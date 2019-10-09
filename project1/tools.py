from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
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

def check_centering(y, yc):
    """ tests that a variable have been properly mean centered"""
    """ Standard deviation should be the same before and after centering"""
    """ Mean of centered variable should be 0 """

    if isinstance(y, (list, np.ndarray)):
        print('|std(y) -std(yc)| : ', np.abs(np.sqrt(np.var(y, axis=0))-np.sqrt(np.var(yc,axis=0))))
        print('mean(y) : ', np.mean(yc,axis=0))

    else:
        print('|std(y) -std(yc)| : ', np.abs(np.sqrt(np.var(y))-np.sqrt(np.var(yc))))
        print('mean(y): ', np.mean(yc))

def check_eq(list1, val): 
    return(all(x == val for x in list1))

def plot_CI(param, param_var, print_CI = False, include_intercept=False):
    """ plots the confidence intervals of the parameters at 95% confidence"""
    """ parameter index vs parameter value                                """

    #param_var[np.abs(param_var)<1e-14]=0
    error = 1.96*np.sqrt(param_var)

    if print_CI:
        print('CI_.95 beta:')
        for i in range(len(param)):
            print('   %.2f +- %.2f' % (param[i], error[i]))

    if include_intercept:
        indices = np.arange(0, len(param_var),1)
    else: 
        indices = np.arange(1, len(param_var)+1,1)

    plt.errorbar(indices, param, yerr=error, fmt='o', capsize=4, capthick=1.5)
    plt.xlabel(r'$\beta$ - parameter index', size=14)
    plt.ylabel(r'$\beta$ value', size=14)
    plt.tick_params(axis='both', labelsize=12)
    
    plt.grid('True', linestyle='--')
    plt.tight_layout()
    plt.show()
