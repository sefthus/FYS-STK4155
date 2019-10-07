
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
from random import random, seed
import sys
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
#import sklearn.linear_model as skl # import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.utils import shuffle

import tools
import regression_methods as regmet
import CV_methods as cvmet

np.random.seed(3155)
#np.random.seed(2018)

def FrankeFunction(x,y):
    """ the Franke function """

    term1 = 0.75 * np.exp(-(0.25 * (9*x-2)**2) - 0.25 * ((9*y-2)**2))
    term2 = 0.75 * np.exp(-((9*x+1)**2) / 49.0 - 0.1 * (9*y+1))
    term3 = 0.5 * np.exp(-(9*x-7)**2 / 4.0 - 0.25 * ((9*y-3)**2))
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def FrankePlot(x, y, z, ztilde=None):
    """ plot 3d surface of z, and possibly also of the z prediction, ztilde"""

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x, y, z, cmap=mpl.cm.coolwarm,
                        linewidth=0, antialiased=False)

    
    if ztilde is not None:
        ax.scatter(x, y, ztilde, alpha=1, s=1, color='black')

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def fit_data(X, z_noise, z_true, stddev=1, lmbda=0, normalize=False, center_var=True, include_intercept=False, test_centering=False, use_sklearn=False, reg_method=LinearRegression, plot_beta=False):

    z_tilde, z_true, beta, varbeta = regmet.best_fit(X, z_noise, z_true, stddev, lmbda, normalize, center_var, include_intercept, test_centering, use_sklearn, reg_method)
    mse_true = tools.MSE_func(z_true, z_tilde)
    mse_noise = tools.MSE_func(z_noise, z_tilde)
    print(' true Mean squared error =', mse_true)
    print(' noise Mean squared error =', mse_noise)
    print(' R2 score           =', tools.R2_score_func(z_true, z_tilde))
    #print(' R2 score sklearn =', r2_score(z1_true,z_tilde)) # same value

    if plot_beta:
        plot_CI(beta, varbeta, include_intercept=True)

def best_fit(X, z_noise, z_true, stddev=1, lmbda=0, normalize=False, center_var = True, include_intercept=False, test_centering = False, use_sklearn=False, reg_method=Ridge):
    """ do a regression analysis without cross validating      """
    """ choose to mean-center variables and test the centering """
    """ chose to use sklearn or not                            """
    #X = create_design_matrix(x, y, d) 

    if center_var: # mean-center X and z

        if check_eq(X[:,0], 1) and not include_intercept:
            X = X[:,1:] # remove intercept if centering
            print('removing intercept column')
        
        if use_sklearn:

            scalerX = StandardScaler(with_std=True).fit(X)
            X_c = scalerX.transform(X)

            scalerz = StandardScaler(with_std=False).fit(z_noise.reshape(-1,1))
            z_noise_c = scalerz.transform(z_noise.reshape(-1,1)).ravel()

            #scalerztrue = StandardScaler(with_std=False).fit(z_true.reshape(-1,1))
            z_true_c = z_true#scalerztrue.transform(z_true.reshape(-1,1)).ravel()

            Idm = np.identity(np.shape(X_c.T)[0]) # identity matrix
            XtXinv = invert_matrix(np.matmul(X_c.T, X_c) + lmbda*Idm)

            if reg_method == LinearRegression:
                model = reg_method(fit_intercept=False)
            else:
                if lmbda == 0:
                    print('use reg_method=LinearRegression instead of lmbda=0')
                    sys.exit()
                model = reg_method(alpha=lmbda, normalize=normalize, fit_intercept=False, max_iter = 1e4,tol = 0.001)

            model.fit(X_c, z_noise_c)

            z_tilde = model.predict(X_c) + scalerz.mean_
            z_noise_mean = scalerz.mean_
            beta = model.coef_
            if include_intercept and not check_eq(X[:,0], 1):
                beta = np.append(np.array([scalerz.mean_]), beta.ravel())

        else:

            X_c = (X - np.mean(X, axis=0))/np.std(X, axis=0)

            Idm = np.identity(np.shape(X_c.T)[0])
            XtXinv = invert_matrix(np.matmul(X_c.T, X_c) + lmbda*Idm)

            z_noise_mean = np.mean(z_noise)
            z_noise_c = z_noise - z_noise_mean

            #z_true_mean = np.mean(z_true)
            z_true_c = z_true# - z_true_mean

            beta = XtXinv.dot(X_c.T).dot(z_noise_c)
            z_tilde = (np.matmul(X_c, beta) + z_noise_mean)
            beta = beta#np.append(z_noise_mean, beta.ravel())
            if include_intercept and not check_eq(X[:,0], 1):
                beta = np.append(z_noise_mean, beta.ravel())

        if test_centering:
            check_centering(X, X_c)
            check_centering(z_true, z_true_c)
            check_centering(z_noise, z_noise_c)
        
        varbeta = stddev**2*np.diag(XtXinv)
        if include_intercept and not check_eq(X[:,0], 1):
            varbeta = np.append(stddev**2*np.var(z_noise)/len(z_noise),varbeta)
    else:

        Idm = np.identity(np.shape(X.T)[0])
        XtXinv = invert_matrix(np.matmul(X.T,X) + lmbda*Idm)
        if use_sklearn:
            if reg_method == LinearRegression:
                model = reg_method(fit_intercept=True)
            else:
                model = reg_method(alpha=lmbda, fit_intercept=True, max_iter = 1e6,tol = 0.001)
            model.fit(X, z_noise)

            z_tilde = model.predict(X)
            beta = model.coef_
        else:
            beta = XtXinv.dot(X.T).dot(z_noise)
            z_tilde = np.matmul(X, beta)
        varbeta = stddev**2*np.diag(XtXinv)
    return z_tilde, z_true, beta, varbeta

def main():

    # Make data.
    print(' make data')
    #x = np.sort(np.random.random(100))
    x = np.arange(0, 1, 0.01)
    #x = np.linspace(0, 1, 50, endpoint=True)
    #sys.exit()
    #y = np.sort(np.random.random(100))
    y = np.arange(0, 1, 0.01)
    #y = np.linspace(0, 1, 50, endpoint=True)
    x, y = np.meshgrid(x, y)

    z = FrankeFunction(x, y)


    print(' turn data matrices to arrays')
    x1 = np.ravel(x)
    y1 = np.ravel(y)
    n = int(len(x1))

    z1_true = np.ravel(z)
    stddev = 1
    z1_noise = np.ravel(z) + np.random.normal(0, stddev, size=z1_true.shape) # adding noise
    #z1_noise = z.ravel() + np.random.randn(n)

    print(' make design matrix')
    X = tools.create_design_matrix(x, y, d=5)

    # ----------- OLS-------------------
    print('--------------- OLS ----------------------')
    print('  -------------- no centering')
    fit_data(X, z1_noise, z1_true, stddev, center_var=False, use_sklearn=True, reg_method=LinearRegression, include_intercept=True)

    print('   --------------- centering manually')
    fit_data(X, z1_noise, z1_true, stddev, center_var=True, use_sklearn=False, reg_method=LinearRegression)
   
    print('   --------------- centering sklearn')
    fit_data(X[:,1:], z1_noise, z1_true, stddev, center_var=True, include_intercept=True, use_sklearn=True, reg_method=LinearRegression)

    print('   ------------- cross validating')
    cvmet.kfold_CV(X[:,1:], z1_noise, z1_true) # same result when using same random_state=0 in Kfold and shuffle
    #cvmet.kfold_CV_sklearn(X[:,1:], z1_noise, z1_true, reg_method=LinearRegression) # same result when using same random_state=0 in Kfold and shuffle
    
    #FrankePlot(x,y,z,z_tilde_cm)
    print('   ---------- finding best fit polynomial')
    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, cv_func=kfold_CV)

    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, cv_func=kfold_CV_sklearn, reg_method=LinearRegression)

    # ------------- Ridge 
    print('--------- Ridge regression --------------')
    #regression_lmbda(x1, y1, z1_noise, z1_true, d=5, cv_method=kfold_CV, reg_method=Ridge, plot_mse=True)
    print('    ------- no cross validation')
    #fit_data(X[:,1:], z1_noise, z1_true, stddev, lmbda=1e-3, center_var=True, include_intercept=False, use_sklearn=False, reg_method=Ridge)
  
    print('    ---------- cross-validating')
    #cvmet.kfold_CV(X[:,1:], z1_noise, z1_true, lmbda=1e-3)

    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, 14,lmbda=1e-3, cv_func=kfold_CV, reg_method=Ridge)
    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, 14,lmbda=1e-3,cv_func=kfold_CV_sklearn, reg_method=Ridge)

    print( '    ----- best fit based on lambda and polynomial degree')
    #ridge_bias_variance(x1, y1, z1_noise, z1_true)
    #regression_lmbda(x1, y1, z1_noise, z1_true, d=5, reg_method=Ridge)

    # ------------- Lasso
    print('---------------- Lasso regression --------------')
    regression_lmbda(x1, y1, z1_noise, z1_true, d=5, cv_method=kfold_CV_sklearn, reg_method=Lasso, plot_mse=True, normalize=True)
    #fit_data(X[:,1:], z1_noise, z1_true, stddev, lmbda=1e-4, center_var=True, include_intercept=True, use_sklearn=True, reg_method=Lasso, plot_beta=False)

    #kfold_CV(X[:,1:], z1_noise, z1_true, lmbda=1e-4)

    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, 14,lmbda=1e-4, cv_func=kfold_CV_sklearn, reg_method=Lasso)
    #lasso_bias_variance(x1, y1, z1_noise, z1_true)
    #regression_lmbda(x1, y1, z1_noise, z1_true, d=5, reg_method=Lasso)
    #plt.show()

def kfold_CV(X, z, z_true, splits = 5, return_var = False, lmbda=0, normalize=None, return_bv=False, reg_method=None):

    kfold = KFold(n_splits=splits, shuffle=True)#, random_state=0)

    mse_splits_test = np.zeros(splits)
    mse_splits_train = np.zeros(splits)
    r2_splits_test = np.zeros(splits)

    bias_splits = np.zeros(splits)
    variance_splits = np.zeros(splits)

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
        XtXinv_train = invert_matrix(np.matmul(X_train_c.T, X_train_c) + lmbda*Idm)
        beta = XtXinv_train.dot(X_train_c.T).dot(z_train_c)

        # evaulate model on test set
        z_tilde_test = np.matmul(X_test_c,beta) + z_train_mean
        z_tilde_train = np.matmul(X_train_c,beta) + z_train_mean


        mse_splits_test[i] = np.mean((z_true_test_c - z_tilde_test)**2)
        mse_splits_train[i] = np.mean((z_true_train_c - z_tilde_train)**2)
        r2_splits_test[i] = R2_score_func(z_true_test_c, z_tilde_test)

        bias_splits[i] = np.mean((z_true_test_c - np.mean(z_tilde_test))**2)
        variance_splits[i] = np.var(z_tilde_test)
        
        i += 1

    # calculate errors, and the bias and variance scores
    mse_test = np.mean(mse_splits_test)
    mse_train = np.mean(mse_splits_train)
    r2_test = np.mean(r2_splits_test)
    bias = np.mean(bias_splits)
    variance = np.mean(variance_splits)
    #print('{} >= {}'.format(mse_test, bias + variance))

    if return_var and return_bv:
        return mse_test, mse_train, r2_test, bias, variance

    if return_var and not return_bv:
        return mse_test, mse_train, r2_test


    print(' CV MSE_scores   :',mse_test)
    print(' CV R2 score        :', r2_test)

def kfold_CV_sklearn(X, z, z_true, splits = 5, normalize=False, return_var = False, lmbda=0, return_bv = False, reg_method=Ridge):
    """ k-fold cross validation using the sklearn library """
    """ design matrix must be without the first column [1,1,...1] """

    kfold = KFold(n_splits=splits, shuffle=True)#, random_state=0)

    mse_splits_test = np.zeros(splits)
    mse_splits_train = np.zeros(splits)
    r2_splits_test = np.zeros(splits)

    bias_splits = np.zeros(splits)
    variance_splits = np.zeros(splits)
    
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
        else:
            model = reg_method(alpha=lmbda, normalize=normalize, fit_intercept=True, max_iter = 1e4,tol = 0.001)
        model.fit(X_train_c, z_train_c)

        # evaulate model on test set
        z_tilde_test = model.predict(X_test_c) + scalerz.mean_
        z_tilde_train = model.predict(X_train_c) + scalerz.mean_


        mse_splits_test[i] = np.mean((z_true_test_c - z_tilde_test)**2)
        mse_splits_train[i] = np.mean((z_true_train_c - z_tilde_train)**2)
        r2_splits_test[i] = R2_score_func(z_true_test_c, z_tilde_test)

        bias_splits[i] = np.mean((z_true_test_c - np.mean(z_tilde_test))**2)
        variance_splits[i] = np.var(z_tilde_test)
        
        i += 1


    # calculate errors, and the bias and variance scores    
    mse_test = np.mean(mse_splits_test)
    mse_train = np.mean(mse_splits_train)
    r2_test = np.mean(r2_splits_test)
    bias = np.mean(bias_splits)
    variance = np.mean(variance_splits)

    if return_var and return_bv:
        return mse_test, mse_train, r2_test, bias, variance

    if return_var and not return_bv:
        return mse_test, mse_train, r2_test
    
    #print('{} >= {}'.format(mse_test, bias + variance))
    print(' CV sklearn MSE_scores:', mse_test)
    print(' CV R2 score sklearn     :', r2_test)
    
def bias_variance_tradeoff(x, y, z_noise, z_true, degree_max = 14, lmbda=0, return_var = False, cv_func = kfold_CV, reg_method=None):

    error_test = np.zeros(degree_max-1)
    error_train = np.zeros(degree_max-1)
    r2_test = np.zeros(degree_max-1)

    bias = np.zeros(degree_max-1)
    variance = np.zeros(degree_max-1)
    vb = np.zeros(degree_max-1)

    polydegree = np.arange(1,degree_max,1)

    for d in range(degree_max-1):

        #print('degree:', d)

        X = create_design_matrix(x, y, d+1)
        mse_deg_test, mse_deg_train, r2_deg_test, bias_deg, variance_deg = cv_func(X[:,1:], z_noise, z_true, return_var=True, lmbda=lmbda, return_bv=True, reg_method=reg_method)

        error_test[d] = mse_deg_test
        error_train[d] = mse_deg_train
        r2_test[d] = r2_deg_test

        bias[d] = bias_deg
        variance[d] = variance_deg
        vb[d] = variance[d] + bias[d]

    if return_var:
        return error_test, error_train, r2_test, bias, variance, vb
    
    print('kfold min MSE    : ', np.min(error_test))
    print('kfold min_err r2 : ', r2_test[np.argmin(error_test)])

    fig = plt.figure()
    plt.plot(polydegree, error_test, label='Test Error')
    plt.plot(polydegree, error_train, '--', label='Train Error')
    #plt.ylim([-0.1, 0.4])
    #plt.xlim([-0.2,13.5])
    #plt.plot(polydegree, bias, '--', label='Bias')
    #plt.plot(polydegree, variance, '--', label='Variance')
    #plt.plot(polydegree, vb, '--', label='Variance+bias')
    #plt.ylim([1.,1.06])
    plt.xlabel('Polynomial order', size=14)
    plt.ylabel('Mean Squared Error', size=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid('True', linestyle='dashed')
    plt.tight_layout()
    plt.legend()
    #fig.savefig("d_MSE_vs_poly_noise.pdf", bbox_inches='tight')
    plt.show()

def regression_lmbda(x, y, z_noise, z_true, d=2, cv_method=kfold_CV_sklearn, reg_method=Ridge, plot_mse=True, normalize=False):
    
    lmbda = np.logspace(-7,6,14)

    error_test = np.zeros(len(lmbda))
    r2_test = np.zeros(len(lmbda))
    error_train = np.zeros(len(lmbda))

    bias = np.zeros(len(lmbda))
    variance = np.zeros(len(lmbda))
    vb = np.zeros(len(lmbda))

    for i in range(len(lmbda)):

        print('lambda:',lmbda[i])
        X = create_design_matrix(x, y, d)
        
        mse_l_test, mse_l_train, r2_l_test, *junk = cv_method(X[:,1:], z_noise, z_true, return_var=True, lmbda=lmbda[i], normalize=normalize, return_bv=True)
        
        error_test[i] = mse_l_test
        error_train[i] = mse_l_train
        r2_test[i] = r2_l_test

        #bias[i] = bias_l
        #variance[i] = variance_l
        #vb[i] = variance[i] + bias[i]

    indx = np.argmin(error_test)
    print('min MSE      :', error_test[indx])
    print('minMSE lambda:', lmbda[indx])
    print('max r2       :', r2_test[indx])
    
    if plot_mse:
        plt.plot(lmbda, error_test, label='Test Error')
        #plt.plot(lmbda, error_train, '--',label='Train Error')
        plt.plot(lmbda[indx], error_test[indx], 'x')

        plt.xlabel(r'$\lambda$', size=14)
        plt.ylabel('Mean squared error', size=14)
        plt.xscale('log')
        plt.grid('True', linestyle='dashed')
        plt.legend()
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.show()

def ridge_bias_variance(x, y, z_noise, z_true, degree_max=14, plot_mse=False, normalize=False):

    lmbda = np.logspace(-7,2,10) 
    polydegree = np.arange(1,degree_max,1)

    min_error = np.zeros_like(lmbda)
    min_r2 = np.zeros_like(lmbda)
    min_degrees = np.zeros_like(polydegree)

    for i in range(len(lmbda)):

        print('             lambda:',lmbda[i])  

        error_test, error_train, r2_test, *junk = bias_variance_tradeoff(x, y, z_noise, z_true, degree_max, lmbda=lmbda[i], return_var = True, normalize=normalize)

        min_r2[i] = np.max(r2_test)
        min_error[i] = np.min(error_test)
        min_degrees[i] = polydegree[np.argmin(error_test)]

        plt.plot(polydegree, error_test, label=(r'$\lambda$=%.2e' %lmbda[i]))

    #plt.plot(polydegree, error_train, label=(r'$\lambda=$%.2e Train Error', lmbda[i]))    
    idx_min = np.argmin(min_error)

    print('min error:', min_error[idx_min])
    print('min error poly:', min_degrees[idx_min])
    print('max r2 score:', min_r2[idx_min])

    if plot_mse:
        plt.plot(min_degrees[idx_min], min_error[idx_min], 'kx')
        plt.xlabel('polynomial degree', size= 14)
        plt.ylabel('Mean squared error', size=14)
        plt.grid('True', linestyle='dashed')
        plt.legend()
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.show()

def lasso_bias_variance(x, y, z_noise, z_true, degree_max=14, normalize=False):

    lmbda = np.logspace(-6,2,9)
    #lmbda[-1] = 0
    N_trials = 1
 
    polydegree = np.arange(1,degree_max,1)
    min_error = np.zeros_like(lmbda)
    min_r2 = np.zeros_like(lmbda)
    min_degrees = np.zeros_like(polydegree)

    for i in range(len(lmbda)):

        print('             lambda:',lmbda[i])  

        error_test, error_train, r2_test, *junk = bias_variance_tradeoff(x, y, z_noise, z_true, degree_max, lmbda=lmbda[i], return_var = True, cv_func = kfold_CV_sklearn, reg_method = Lasso, normalize=normalize)

        min_r2[i] = np.max(r2_test)
        min_error[i] = np.min(error_test)
        min_degrees[i] = polydegree[np.argmin(error_test)]
        plt.plot(polydegree, error_test, label=(r'$\lambda$=%.2e' %lmbda[i]))

    #plt.plot(polydegree, error_train, label=(r'$\lambda=$%.2e Train Error', lmbda[i]))
    idx_min = np.argmin(min_error)
    print('min error:', min_error[idx_min])
    print('min error poly:', min_degrees[idx_min])
    print('max r2 score:', min_r2[idx_min])

    plt.plot(min_degrees[idx_min], min_error[idx_min], 'x')
    plt.xlabel('polynomial degree', size=14)
    plt.ylabel('Mean squared error', size=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.grid('True', linestyle='dashed')
    plt.tight_layout()
    plt.legend()
    plt.show()

main()