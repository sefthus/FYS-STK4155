
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
    bias_variance_tradeoff(x1, y1, z1_noise, z1_true, cv_func=cvmet.kfold_CV)

    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, cv_func=cvmet.kfold_CV_sklearn, reg_method=LinearRegression)

    # ------------- Ridge 
    print('--------- Ridge regression --------------')
    regression_lmbda(x1, y1, z1_noise, z1_true, d=5, cv_method=cvmet.kfold_CV, reg_method=Ridge, plot_mse=True)
    print('    ------- no cross validation')
    fit_data(X[:,1:], z1_noise, z1_true, stddev, lmbda=1e-3, center_var=True, include_intercept=False, use_sklearn=False, reg_method=Ridge, plot_beta=True, normalize=False)
    sys.exit()
    print('    ---------- cross-validating')
    #cvmet.kfold_CV(X[:,1:], z1_noise, z1_true, lmbda=1e-3)

    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, 14,lmbda=1e-3, cv_func=kfold_CV, reg_method=Ridge)
    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, 14,lmbda=1e-3,cv_func=kfold_CV_sklearn, reg_method=Ridge)

    print( '    ----- best fit based on lambda and polynomial degree')
    #ridge_bias_variance(x1, y1, z1_noise, z1_true)
    #regression_lmbda(x1, y1, z1_noise, z1_true, d=5, reg_method=Ridge)

    # ------------- Lasso
    print('---------------- Lasso regression --------------')
    regression_lmbda(x1, y1, z1_noise, z1_true, d=5, cv_method=cvmet.kfold_CV_sklearn, reg_method=Lasso, plot_mse=True, normalize=True)
    #fit_data(X[:,1:], z1_noise, z1_true, stddev, lmbda=1e-4, center_var=True, include_intercept=True, use_sklearn=True, reg_method=Lasso, plot_beta=False)

    #kfold_CV(X[:,1:], z1_noise, z1_true, lmbda=1e-4)

    #bias_variance_tradeoff(x1, y1, z1_noise, z1_true, 14,lmbda=1e-4, cv_func=kfold_CV_sklearn, reg_method=Lasso)
    #lasso_bias_variance(x1, y1, z1_noise, z1_true)
    #regression_lmbda(x1, y1, z1_noise, z1_true, d=5, reg_method=Lasso)
    #plt.show()
    
def bias_variance_tradeoff(x, y, z_noise, z_true, degree_max = 14, lmbda=0, return_var = False, cv_func = cvmet.kfold_CV, reg_method=None):

    error_test = np.zeros(degree_max-1)
    error_train = np.zeros(degree_max-1)
    r2_test = np.zeros(degree_max-1)

    bias = np.zeros(degree_max-1)
    variance = np.zeros(degree_max-1)
    vb = np.zeros(degree_max-1)

    polydegree = np.arange(1,degree_max,1)

    for d in range(degree_max-1):

        #print('degree:', d)

        X = tools.create_design_matrix(x, y, d+1)
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

def regression_lmbda(x, y, z_noise, z_true, d=2, cv_method=cvmet.kfold_CV_sklearn, reg_method=Ridge, plot_mse=True, normalize=False):
    
    lmbda = np.logspace(-7,6,14)

    error_test = np.zeros(len(lmbda))
    r2_test = np.zeros(len(lmbda))
    error_train = np.zeros(len(lmbda))

    bias = np.zeros(len(lmbda))
    variance = np.zeros(len(lmbda))
    vb = np.zeros(len(lmbda))

    for i in range(len(lmbda)):

        print('lambda:',lmbda[i])
        X = tools.create_design_matrix(x, y, d)
        
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

        error_test, error_train, r2_test, *junk = bias_variance_tradeoff(x, y, z_noise, z_true, degree_max, lmbda=lmbda[i], return_var = True, cv_func = cvmet.kfold_CV_sklearn, reg_method = Lasso, normalize=normalize)

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