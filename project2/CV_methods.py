#from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
#import sklearn.linear_model as skl # import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

import tools

def kfold_CV(X, z, z_true, splits = 5, return_var = False, lmbda=0, return_beta_var=False, return_bv=False, franke_plot=False, reg_method=None):
    """ 
      k-fold cross validation                                   
      design matrix must be without the first column [1,1,...1] 
    """

    kfold = KFold(n_splits=splits, shuffle=True)#, random_state=0)

    mse_splits_test = np.zeros(splits)
    mse_splits_train = np.zeros(splits)
    r2_splits_test = np.zeros(splits)

    #z_tilde_splits_test = np.zeros((splits, X.shape[0]/splits))
    #z_tilde_splits_train = np.zeros((splits, X.shape[0]*(1-splits)/splits))

    z_tilde_splits_test = np.zeros(splits, dtype=np.ndarray)
    z_tilde_splits_train = np.zeros(splits, dtype=np.ndarray)

    betas = np.zeros((splits, X.shape[1]))
    var_beta = np.zeros((splits, X.shape[1] ))

    i=0
    for train_idx, test_idx in kfold.split(X):

        X_train = X[train_idx]
        z_train = z[train_idx]
        z_true_train = z_true[train_idx]

        X_train_mean = np.mean(X_train, axis=0)
        X_train_c = (X_train - X_train_mean)/np.std(X_train, axis=0)
        
        z_train_mean = np.mean(z_train)
        z_train_c = z_train - z_train_mean

        #z_true_train_mean = np.mean(z_true_train)
        z_true_train_c = z_true_train# - z_true_train_mean


        X_test = X[test_idx]
        X_test_c = (X_test - X_train_mean)/np.std(X_train, axis=0)
        
        z_test = z[test_idx]
        z_test_c = z_test - z_train_mean
        z_true_test_c = z_true[test_idx]# - z_true_train_mean

        # fit model on training set
        Idm = np.identity(np.shape(X_train_c.T)[0])
        XtXinv_train = tools.invert_matrix(np.matmul(X_train_c.T, X_train_c) + lmbda*Idm)
        beta = XtXinv_train.dot(X_train_c.T).dot(z_train_c)

        # evaulate model on test set
        z_tilde_splits_test[i] = np.matmul(X_test_c,beta) + z_train_mean
        z_tilde_splits_train[i] = np.matmul(X_train_c,beta) + z_train_mean
        betas[i,:] = beta

        mse_splits_test[i] = np.mean((z_true_test_c - z_tilde_splits_test[i])**2)
        mse_splits_train[i] = np.mean((z_true_train_c - z_tilde_splits_train[i])**2)
        r2_splits_test[i] = tools.R2_score_func(z_true_test_c, z_tilde_splits_test[i])

        
        i += 1

    # calculate errors, and the bias and variance scores
    mse_test = np.mean(mse_splits_test)
    mse_train = np.mean(mse_splits_train)
    r2_test = np.mean(r2_splits_test)

    z_tilde_test = np.array([np.mean(z_tilde) for z_tilde in z_tilde_splits_test])
    z_tilde_train = np.array([np.mean(z_tilde) for z_tilde in z_tilde_splits_train])   


    if return_beta_var:
        return np.mean(betas, axis=0), np.mean(var_beta, axis=0)#np.var(betas, axis=0)/(splits-1)

    else:
        return z_tilde_test, z_tilde_train, mse_test, mse_train, r2_test
    #print(' CV MSE_scores   :', mse_test)
    #print(' CV R2 score        :', r2_test)

def kfold_CV_sklearn(X, z, z_true, splits = 5, return_var = False, lmbda=0, return_bv = False, return_beta_var=False, reg_method=Ridge):
    """ 
        k-fold cross validation using the sklearn library
        matrix X must be without the first intercept column [1,1,...1] 
    """

    kfold = KFold(n_splits=splits, shuffle=True)#, random_state=0)

    mse_splits_test = np.zeros(splits)
    mse_splits_train = np.zeros(splits)
    r2_splits_test = np.zeros(splits)

    #z_tilde_splits_test = np.zeros((splits, X.shape[0]/splits))
    #z_tilde_splits_train = np.zeros((splits, X.shape[0]*(splits-1)/splits))

    z_tilde_splits_test = np.zeros(splits, dtype=np.ndarray)
    z_tilde_splits_train = np.zeros(splits, dtype=np.ndarray)

    betas = np.zeros((splits, X.shape[1]))

    i=0
    for train_idx, test_idx in kfold.split(X):

        X_train = X[train_idx]
        scalerX = StandardScaler(with_std = True).fit(X_train)
        X_train_c = scalerX.transform(X_train)

        z_train = z[train_idx]
        scalerz = StandardScaler(with_std=False).fit(z_train.reshape(-1,1))
        z_train_c = scalerz.transform(z_train.reshape(-1,1)).ravel()

        z_true_train = z_true[train_idx]
        #scalerztrue = StandardScaler(with_std=False).fit(z_true_train.reshape(-1,1))
        z_true_train_c = z_true_train#scalerztrue.transform(z_true_train.reshape(-1,1)).ravel()


        X_test_c = scalerX.transform(X[test_idx])
        z_test_c = scalerz.transform(z[test_idx].reshape(-1,1)).ravel()
        z_true_test_c = z_true[test_idx]#scalerztrue.transform(z_true[test_idx].reshape(-1,1)).ravel()

        # fit model on training set
        if reg_method == LinearRegression:
            model = reg_method(fit_intercept=True)
        elif reg_method == Ridge:
            model = reg_method(alpha=lmbda, max_iter = 1e5, tol = 0.001)
        elif reg_method == Lasso:
            model = reg_method(alpha=lmbda, precompute=True, max_iter = 1e6, tol = 0.001)
        model.fit(X_train_c, z_train_c)
        betas[i,:] = model.coef_
        # evaulate model on test set
        z_tilde_splits_test[i] = model.predict(X_test_c) + scalerz.mean_
        z_tilde_splits_train[i] = model.predict(X_train_c) + scalerz.mean_


        mse_splits_test[i] = np.mean((z_true_test_c - z_tilde_splits_test[i])**2)
        mse_splits_train[i] = np.mean((z_true_train_c - z_tilde_splits_train[i])**2)
        r2_splits_test[i] = tools.R2_score_func(z_true_test_c, z_tilde_splits_test[i])


        
        i += 1


    # calculate errors, and the bias and variance scores    
    mse_test = np.mean(mse_splits_test)
    mse_train = np.mean(mse_splits_train)
    r2_test = np.mean(r2_splits_test)

    z_tilde_test = np.array([np.mean(z_tilde) for z_tilde in z_tilde_splits_test])
    z_tilde_train = np.array([np.mean(z_tilde) for z_tilde in z_tilde_splits_train])   


    return z_tilde_test, z_tilde_train, mse_test, mse_train, r2_test

    #print('{} >= {}'.format(mse_test, bias + variance))
    #print(' CV sklearn MSE_scores:', mse_test)
    #print(' CV R2 score sklearn     :', r2_test)