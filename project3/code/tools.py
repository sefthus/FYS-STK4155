from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
import seaborn as sns
from random import random, seed
import sys

from sklearn.preprocessing import StandardScaler

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
    
    return X[:,1:]

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

    """ calculates the r2 score """

    return 1 - (np.sum((ytrue-ypredict)**2)/np.sum((ytrue-np.mean(ytrue))**2))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def scale_data(X_train, X_test, y_train = None, y_test = None, scale_y=False, cat_split=None, turn_dense=False):
    """
        Scales the training and test data with StandardScaler() from scikit learn.
        
        Classification case: The categorical data should already be onehotencoded.
            The data is first split into numerical and categorical data according to
            cat_split, and then the numerical data is scaled.
        
        Regression case: cat_split must be set to None, and y_traina and y_test must be input
            In addition to the design matrix X, the response y is also scaled.
        
        Returns the scaled training and test data. If cat_split=None, the scaled response is
        also returned.

        X_train:    the training data
        X_test:     the test data
        y_train:    the training response. Set to None for classification
        y_test:     the test response. Set to None for classification
        scale_y:    scales the response training and test data as well. y_train and y_test can not be None.
        cat_split:  where to split the data into categorical and numerical arrays
        turn_dense: if the data is a sparse matrix, setting this to True, 
                    will transform the data to dense numpy arrays 
    """
    sparse_array = False

    if sparse.issparse(X_train):
        X_train, X_test = X_train.toarray(), X_test.toarray()
        sparse_array = True

    if cat_split is None:
        X_train_num = X_train
        X_test_num = X_test

    else:
        X_train_cat, X_train_num = X_train[:, :cat_split], X_train[:, cat_split:]
        X_test_cat, X_test_num = X_test[:, :cat_split], X_test[:, cat_split:]

    scaleX = StandardScaler().fit(X_train_num)
    
    X_train_scaled = scaleX.transform(X_train_num)
    X_test_scaled = scaleX.transform(X_test_num)

    if cat_split is not None:
        X_train_scaled = np.concatenate((X_train_cat, X_train_scaled), axis=1)
        X_test_scaled = np.concatenate((X_test_cat, X_test_scaled), axis=1)

    if sparse_array:
        if not turn_dense:
            X_train_scaled = sparse.csr_matrix(X_train_scaled)
            X_test_scaled = sparse.csr_matrix(X_test_scaled)
        
    if scale_y:
        scaley = StandardScaler(with_std=False).fit(y_train)
        y_train = scaley.transform(y_train)
        y_test = scaley.transform(y_test)

        return X_train, X_test, y_train, y_test, scaley.mean_

    else:
        return X_train, X_test


def check_centering(y, yc):
    """ 
        tests that a variable have been properly mean centered
        Standard deviation should be the same before and after centering
        Mean of centered variable should be 0 
    """

    if isinstance(y, (list, np.ndarray)):
        print('|std(y) -std(yc)| : ', np.abs(np.sqrt(np.var(y, axis=0))-np.sqrt(np.var(yc,axis=0))))
        print('mean(y) : ', np.mean(yc,axis=0))

    else:
        print('|std(y) -std(yc)| : ', np.abs(np.sqrt(np.var(y))-np.sqrt(np.var(yc))))
        print('mean(y): ', np.mean(yc))

def check_eq(list1, val): 
    return(all(x == val for x in list1))

def drop_value(dframe, variables, val, drop_less_than=False):
    
    """ 
        drop rows which have values equal to val

        dframe: data frame
        vars  : data frame variables, i.e., dframe.PAY_0
        val   : value to be dropped
        
    """

    if drop_less_than:
        dframe = dframe.drop(dframe[(variables<val)].index)
    else:
        dframe = dframe.drop(dframe[(variables == val)].index)

    return dframe

def bestCurve(y):
    """ creates the best curve in a cumulative gains chart from input y """
    defaults = np.sum(y == 1)
    total = len(y)
    x = np.linspace(0, 1, total)
    y1 = np.linspace(0, 1, defaults)
    y2 = np.ones(total-defaults)
    y3 = np.concatenate([y1,y2])
    return x, y3

def vizualize_scores(metric_score, title=' ', xticks=None, yticks=None, xlab='param 1', ylab='param 2'):
        fig, ax = plt.subplots(figsize = (8, 8))
        #sns.heatmap(test_accuracy, xticklabels=lmbd_vals, yticklabels=eta_vals, annot=True, ax=ax, cmap="viridis")
        #ax.set_title("Test Accuracy")
    
        sns.heatmap(metric_score, xticklabels=xticks, yticklabels=xticks, annot=True, ax=ax, cmap="viridis")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()