from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
import sys
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
#import sklearn.linear_model as skl # import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

np.random.seed(2018)
def bias_variance_tradeoff(degree_max=14):
    n = 40

    x_trial = np.linspace(-3, 3, n).reshape(-1, 1)
    z = np.exp(-x_trial**2) + 1.5 * np.exp(-(x_trial-2)**2)+ np.random.normal(0, 0.1, x_trial.shape)
    z_true = np.exp(-x_trial**2) + 1.5 * np.exp(-(x_trial-2)**2)+ np.random.normal(0, 0.1, x_trial.shape)

    error = np.zeros(degree_max)
    bias = np.zeros(degree_max)
    variance = np.zeros(degree_max)
    vb = np.zeros(degree_max)

    polydegree = np.arange(0,degree_max,1)

    splits = 5
    kfold = KFold(n_splits=splits, shuffle=True)#, random_state=0)
    clf = LinearRegression(fit_intercept=False) 

    for d in range(degree_max):
        
        print('degree=',d)
        
        poly = PolynomialFeatures(degree=d)
        X = poly.fit_transform(x_trial)

        i=0
        z_tilde = np.zeros((len(z)//splits, splits))

        mse_splits = np.zeros(splits)
        bias_splits = np.zeros(splits)
        variance_splits = np.zeros(splits)

        for train_idx, test_idx in kfold.split(X):
            
            X_train = X[train_idx]
            z_train = z[train_idx]
            z_true_train = z_true[train_idx]

            X_test = X[test_idx]
            z_test = z[test_idx]
            z_true_test= z_true[test_idx]

            z_tilde[:,i] = clf.fit(X_train, z_train).predict(X_test).ravel()#X_test.dot(beta).ravel()

            mse_splits[i] = np.mean((z_true_test - z_tilde[:,i])**2)
            bias_splits[i] = np.mean((z_true_test - np.mean(z_tilde[:,i]))**2)
            variance_splits[i] = np.var(z_tilde[:,i])
            i += 1
            #sys.exit()
        
        error[d]= np.mean(mse_splits)
        bias[d] = np.mean(bias_splits)
        variance[d] = np.mean(variance_splits)
        vb[d] = variance[d]+bias[d]

        print('variance:',variance[d])
        print('bias:', bias[d])
        print('error:', error[d])
        print('{} >= {}'.format(error[d], bias[d]+variance[d]))

        #sys.exit()

    plt.plot(polydegree, error, label='Error')
    #plt.ylim([-0.1, 0.4])
    #plt.xlim([-0.2,13.5])
    plt.plot(polydegree, bias, '--', label='Bias')
    plt.plot(polydegree, variance, '--',label='Variance')
    plt.plot(polydegree, vb, '--',label='Variance+bias')

    plt.legend()
    plt.show()

bias_variance_tradeoff()


def bias_variance_tradeoff1(x, y, z, z_true, degree_max=14):

    x = x.reshape((-1,1))
    y = y.reshape((-1,1))
    z = z.reshape((-1,1))
    z_true = z_true.reshape((-1,1))

    error_test = np.zeros(degree_max)
    error_train = np.zeros(degree_max)
    bias = np.zeros(degree_max)
    variance = np.zeros(degree_max)
    vb = np.zeros(degree_max)

    polydegree = np.arange(0,degree_max,1)

    splits = 5
    kfold = KFold(n_splits=splits, shuffle=True)#, random_state=0)
    clf = LinearRegression(fit_intercept=False) 

    for d in range(degree_max):
        print('degree=',d)
        
        #X = create_design_matrix(x, y, d)
        poly = PolynomialFeatures(degree=d)
        X = poly.fit_transform(np.column_stack((x,y)))
        #print('X:\n',X)
        i=0

        mse_splits_test = np.zeros(splits)
        mse_splits_train = np.zeros(splits)
        bias_splits = np.zeros(splits)
        variance_splits = np.zeros(splits)
        
        for train_idx, test_idx in kfold.split(X):
            
            X_train = X[train_idx]
            z_train = z[train_idx]
            z_true_train = z_true[train_idx]

            X_test = X[test_idx]
            z_test = z[test_idx]
            z_true_test= z_true[test_idx]

           # XtXinv_train = invert_matrix(np.matmul(X_train.T, X_train))
            # evaulate model on test set
            #beta = XtXinv_train.dot(X_train.T).dot(z_train)
            #z_tilde = X_test.dot(beta).ravel()

            z_tilde_test = clf.fit(X_train, z_train).predict(X_test)
            
            z_tilde_train = clf.fit(X_train, z_train).predict(X_train)
            #z_tilde = X_test.dot((clf.coef_).ravel())

            mse_splits_test[i] = np.mean((z_test - z_tilde_test)**2)
            mse_splits_train[i] = np.mean((z_train - z_tilde_train)**2)
            bias_splits[i] = np.mean((z_true_test - np.mean(z_tilde_test))**2)
            variance_splits[i] = np.var(z_tilde_test)

            i += 1
        #print('beta=', beta.ravel())
        #print('coef:', clf.coef_)    
        
        error_test[d]= np.mean(mse_splits_test)
        error_train[d]= np.mean(mse_splits_train)

        bias[d] = np.mean(bias_splits)
        variance[d] = np.mean(variance_splits)
        vb[d] = variance[d]+bias[d]


        #print('variance:',variance[d])
        #print('bias:', bias[d])
        #print('error:', error[d])
        print('{} >= {}'.format(error_test[d], bias[d]+variance[d]))

        #sys.exit()

    plt.plot(polydegree, error_test, label='Test Error')
    plt.plot(polydegree, error_train, label='Train Error')
    #plt.ylim([-0.1, 0.4])
    #plt.xlim([-0.2,13.5])
    #plt.plot(polydegree, bias, '--', label='Bias')
    #plt.plot(polydegree, variance, '--',label='Variance')
    #plt.plot(polydegree, vb, '--',label='Variance+bias')
    #plt.plot(polydegree, error-vb, '--', label='error-(v+b)')

    plt.legend()
    plt.show()
