
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

from scipy.misc import imread, face
import scipy.ndimage as ndimage
import tools
import regression_methods as regmet
import CV_methods as cvmet

np.random.seed(3155)
#np.random.seed(2018)

def FrankeFunction(x,y):
    """ 
    calculates the Franke function from x and y from meshgrid
    """

    term1 = 0.75 * np.exp(-(0.25 * (9*x-2)**2) - 0.25 * ((9*y-2)**2))
    term2 = 0.75 * np.exp(-((9*x+1)**2) / 49.0 - 0.1 * (9*y+1))
    term3 = 0.5 * np.exp(-(9*x-7)**2 / 4.0 - 0.25 * ((9*y-3)**2))
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def FrankePlot(x, y, z, ztilde=None):
    """ 
    plot 3d surface of x,y and z, and possibly also of the z prediction, ztilde
    """

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

def terrain_data(skip_x=4, skip_y=4, plot_terrain=False, skip=False, cut_im=True, cut_x=1000, cut_y=1000):
    terrain = imread('data\SRTM_data_Norway_1.tif')
    #if plot_terrain:
    #    plt.figure()
    #    plt.title('Terrain over Norway 1')
    #    plt.imshow(terrain, cmap='gray')
    #    plt.xlabel('X')
    #    plt.ylabel('Y')
    #    plt.show()
    if cut_im:
        print(' cutting the image to 0:%i, 0:%i ' %(cut_y, cut_x))
        terrain = terrain[:cut_y, :cut_x]
        if plot_terrain:
            plt.figure()
            #plt.title('Terrain over Norway 1')
            plt.imshow(terrain, cmap='gray')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.show()
    if skip:
        print(' skipping every %i, %i pixels' %(skip_y, skip_x))
        terrain = terrain[::skip_y, ::skip_x]
    if plot_terrain:
        plt.figure()
        #plt.title('Terrain over Norway 1')
        plt.imshow(terrain, cmap='gray')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()

    return terrain


def main(d=5, degree_max=14, lmbda=1e-3, lmbda_max=1e2, lmbda_min=1e-6, stddev=1, calc_intercept=False, plot_intercept=False, FrankePlot=False, plot_beta=False, plot_mse=True, OLSreg=False, Ridgereg=False, Lassoreg=False, bv_trade=False, find_l_d=False, find_l=False, use_sklearn=False, CV=True):
    """

     Main function to call on to perform linear regression

     d (5)             : polynomial order of design matrix                               
     degree_max (14)   : max polynomial order to be used in bias_variance_tradeoff       
     lmbda (1e-3)      : value of penalization coefficient in Ridge and Lasso regression 
     stddev (1)        : standard deviation to the the noise term epsilon                
     plot_intercept (False) : wheter to include the intercept when plotting. Only set to true if it is calculated 
     FrankePlot(False) : plot the Franke function                                
     plot_beta(False)  : find the cross-validated beta coefficients and plot them
     plot_mse(True)    : plot the mean square error from bias_variance_tradeoff 
     OLSreg(True)      : use Ordinary Least Square regression 
     Ridgereg(False)   : use Ridge regression
     Lassoreg(False)   : use Lasso regression
     bv_trade(False)   : calculate MSE for several d for a given lambda, using bias_variance_tradeoff
     find_l_d(False)   : calculate MSE for several d=[0,max_degree] and several lambda=[lmbda_min,lmbda_max]
     use_sklearn(False): use sklearns algorithm for doing a regression analysis
     CV(True)          : perform a crossvalidation for lambda and d

     """
    
    
    # Make data.
    print(' make data')
    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)

    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y)
    if FrankePlot:
        FrankePlot(x,y,z)

    print(' turn data matrices to arrays')
    x1 = np.ravel(x)
    y1 = np.ravel(y)
    n = int(len(x1))

    z1_true = np.ravel(z)
    stddev = stddev
    z1_noise = np.ravel(z) + np.random.normal(0, stddev, size=z1_true.shape) # adding noise

    print(' make design matrix')
    X = tools.create_design_matrix(x, y, d=d)
    if not calc_intercept:
        X = X[:, 1:]
    
    print('X_shape:', X.shape)
    if use_sklearn:
        cv_method = cvmet.kfold_CV_sklearn
    else: cv_method = cvmet.kfold_CV

    
    if OLSreg: # Use linear regression
        print('   -------------OLS  cross validating')
        cv_method(X, z1_noise, z1_true, reg_method=LinearRegression) # same result when using same random_state=0 in Kfold and shuffle
   
    
        if plot_beta: # plot cross-validated beta coeffs
            beta, varbeta = cv_method(X, z1_noise, z1_true, reg_method=LinearRegression, return_beta_var=True) # same result when using same random_state=0 in Kfold and shuffle
            tools.plot_CI(beta,varbeta, include_intercept=False)
        
    
        print('   ---------- finding best fit polynomial')
        if bv_trade: # find MSE as a function of polynomial order d
            bias_variance_tradeoff(x1, y1, z1_noise, z1_true, degree_max, cv_method=cv_method, reg_method=LinearRegression)

    
    # ------------- Ridge
    if Ridgereg:
        if calc_intercept:
            X=X[:,1:]
        print('--------- Ridge regression --------------')
        if find_l: # find MSE as a function of one polynomial order d and lambda
            regression_lmbda(x1, y1, z1_noise, z1_true, d, lmbda_min, lmbda_max, n_lmbda, cv_method=cv_method, reg_method=Ridge, plot_mse=True)
        print('    ------- no cross validation')
        
        if CV: # run a cross validation
            print('    ---------- cross-validating')
            cvmet.kfold_CV(X, z1_noise, z1_true, lmbda=lmbda, reg_method=Ridge) # same result when using same random_state=0 in Kfold and shuffle

        if plot_beta: # plot beta confidence intervals
            beta, varbeta = cvmet.kfold_CV(X, z1_noise, z1_true, lmbda=1e-2, return_beta_var=True)
            tools.plot_CI(beta,varbeta)

        if bv_trade: # find MSE as a function of polynomial order up to max_degree and lambda
            bias_variance_tradeoff(x1, y1, z1_noise, z1_true, degree_max, lmbda=lmbda, cv_method=cv_method, reg_method=Ridge)
        
        if find_l_d:
            print( '    ----- best fit based on lambda and polynomial degree')
            ridge_bias_variance(x1, y1, z1_noise, z1_true, degree_max, lmbda_min, lmbda_max, n_lmbda, cv_method=cv_method, plot_mse=True)

    # ------------- Lasso
    if Lassoreg:
        if calc_intercept:
            X=X[:,1:]
        print('---------------- Lasso regression --------------')
        if find_l:
            regression_lmbda(x1, y1, z1_noise, z1_true, d, lmbda_min, lmbda_max, n_lmbda, cv_method=cvmet.kfold_CV_sklearn, reg_method=Lasso, plot_mse=True, normalize=False)
        
        if CV:   
            cvmet.kfold_CV_sklearn(X, z1_noise, z1_true, lmbda=lmbda, reg_method=Lasso) # same result when using same random_state=0 in Kfold and shuffle

        if plot_beta:
            beta, varbeta = cvmet.kfold_CV_sklearn(X[:,1:], z1_noise, z1_true, lmbda=lmbda, reg_method=Lasso, return_beta_var=True)
            tools.plot_CI(beta,varbeta)

        if bv_trade:
            bias_variance_tradeoff(x1, y1, z1_noise, z1_true, d, lmbda, cv_method=cvmet.kfold_CV_sklearn, reg_method=Lasso)

        if find_l_d:
            lasso_bias_variance(x1, y1, z1_noise, degree_max, lmbda_min, lmbda_max, n_lmbdaz1_true, plot_mse=plot_mse, normalize=False)
    
def main_terrain(skip_x=40, skip_y=40, d=10, degree_max=20, lmbda=1e-3, lmbda_min=-7, lmbda_max=2, n_lmbda=10, use_sklearn=False, plot_terrain=False, bv_trade=False, find_l=False, find_l_d=False, plot_beta=False, plot_mse=True, OLSreg=False, Ridgereg=False, Lassoreg=False):

    z_mesh = terrain_data(skip_x, skip_y, skip=True, plot_terrain=plot_terrain, cut_im=True)
    print('z_shape', z_mesh.shape)

    x = np.linspace(0, z_mesh.shape[1], z_mesh.shape[1])
    y = np.linspace(0,z_mesh.shape[0], z_mesh.shape[0])  
    x_mesh, y_mesh = np.meshgrid(x,y)

    x1 = x_mesh.ravel()
    y1 = y_mesh.ravel()
    z1 = z_mesh.ravel()
    
    stddev = np.std(z1)

    X = tools.create_design_matrix(x1, y1, d)
    X = X[:,1:]
    
    print('X shape:', X.shape)

    if use_sklearn:
        cv_method=cvmet.kfold_CV_sklearn
    else:
        cv_method=cvmet.kfold_CV
    

    if OLSreg:
        print(' ------------ OLS')
        if bv_trade:
            print(' -------finding best fit polynomial')
            bias_variance_tradeoff(x1, y1, z1, z1, degree_max, lmbda=0, cv_method=cv_method, reg_method=LinearRegression)   

        if plot_beta:
            beta, beta_var = cvmet.kfold_CV(X, z1, z1, reg_method=LinearRegression, return_beta_var=True)
            tools.plot_CI(beta, beta_var, print_CI=False)

    if Ridgereg:
        print(' ------------- Ridge')
        if bv_trade:
            bias_variance_tradeoff(x1, y1, z1, z1, degree_max, lmbda, cv_method=cv_method, reg_method=Ridge)

        if find_l:
            regression_lmbda(x1, y1, z1, z1, d, lmbda_min, lmbda_max, n_lmbda, cv_method, reg_method=Ridge, plot_mse=plot_mse)
        if find_l_d:
            ridge_bias_variance(x1, y1, z1, z1, degree_max, lmbda_min, lmbda_max, n_lmbda, cv_method=cv_method, plot_mse=plot_mse)

    if Lassoreg:
        print(' -------------- Lasso')
        if bv_trade:
            bias_variance_tradeoff(x1, y1, z1, z1, degree_max, lmbda, cv_method=cvmet.kfold_CV_sklearn, reg_method=Lasso)
        if find_l:
            regression_lmbda(x1, y1, z1, z1, d, lmbda_min, lmbda_max, n_lmbda, cv_method=cvmet.kfold_CV, reg_method=Lasso, plot_mse=plot_mse)
        if find_l_d:
            lasso_bias_variance(x1, y1, z1, z1, degree_max, lmbda_min, lmbda_max, n_lmbda, plot_mse)


def bias_variance_tradeoff(x, y, z_noise, z_true, degree_max = 14, lmbda=0, return_var = False, cv_method = cvmet.kfold_CV, reg_method=None, normalize=False):

    error_test = np.zeros(degree_max-1)
    error_train = np.zeros(degree_max-1)
    r2_test = np.zeros(degree_max-1)

    bias = np.zeros(degree_max-1)
    variance = np.zeros(degree_max-1)
    vb = np.zeros(degree_max-1)

    polydegree = np.arange(1,degree_max,1)

    for d in range(degree_max-1):

        print('degree:', d+1)

        X = tools.create_design_matrix(x, y, d+1)
        mse_deg_test, mse_deg_train, r2_deg_test, bias_deg, variance_deg = cv_method(X[:,1:], z_noise, z_true, return_var=True, lmbda=lmbda, normalize=normalize, return_bv=True, reg_method=reg_method)

        error_test[d] = mse_deg_test
        error_train[d] = mse_deg_train
        r2_test[d] = r2_deg_test

        #print('err_test:', error_test[d])
        bias[d] = bias_deg
        variance[d] = variance_deg
        vb[d] = variance[d] + bias[d]

    if return_var:
        return error_test, error_train, r2_test, bias, variance, vb
    
    indx = np.argmin(error_test)
    print('kfold min MSE degree: ', polydegree[indx])
    print('kfold min MSE    : ', np.min(error_test))
    print('kfold min_err r2 : ', r2_test[indx])

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

def regression_lmbda(x, y, z_noise, z_true, d=5, lmbda_min=-6,lmbda_max=2, n_lmbda=9, cv_method=cvmet.kfold_CV_sklearn, reg_method=Ridge, plot_mse=True, normalize=False):
    
    lmbda = np.logspace(lmbda_min, lmbda_max, n_lmbda)
    #lmbda = 10**np.arange(-8,1,0.5)

    error_test = np.zeros(len(lmbda))
    r2_test = np.zeros(len(lmbda))
    error_train = np.zeros(len(lmbda))

    bias = np.zeros(len(lmbda))
    variance = np.zeros(len(lmbda))
    vb = np.zeros(len(lmbda))
    
    X = tools.create_design_matrix(x, y, d)
    for i in range(len(lmbda)):

        print('lambda:',lmbda[i])

        mse_l_test, mse_l_train, r2_l_test, *junk = cv_method(X[:,1:], z_noise, z_true, return_var=True, lmbda=lmbda[i], normalize=normalize, reg_method=reg_method, return_bv=True)
        
        error_test[i] = mse_l_test
        error_train[i] = mse_l_train
        r2_test[i] = r2_l_test

        print(error_test[i])
        #bias[i] = bias_l
        #variance[i] = variance_l
        #vb[i] = variance[i] + bias[i]

    indx = np.argmin(error_test)
    print('min MSE      :', error_test[indx])
    print('minMSE lambda:', lmbda[indx])
    print('max r2       :', r2_test[indx])
    
    #mse_te, mse_tr, r2_te, *junk = cv_method(X[:,1:], z_noise, z_true, return_var=True, lmbda=1e-3, normalize=normalize, return_bv=True)
    #print('min MSE      :', mse_te)
    #print('max r2       :', r2_te)

    if plot_mse:
        plt.plot(lmbda, error_test, label='Test Error')
        plt.plot(lmbda, error_train, '--',label='Train Error')
        plt.plot(lmbda[indx], error_test[indx], 'x')

        plt.xlabel(r'$\lambda$', size=14)
        plt.ylabel('Mean squared error', size=14)
        plt.xscale('log')
        plt.grid('True', linestyle='dashed')
        plt.legend()
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.show()

def ridge_bias_variance(x, y, z_noise, z_true, degree_max=14, lmbda_min=-6, lmbda_max=2, n_lmbda = 20, cv_method=cvmet.kfold_CV, plot_mse=False, normalize=False):

    lmbda = np.logspace(lmbda_min,lmbda_max,n_lmbda) 
    polydegree = np.arange(1,degree_max,1)

    min_error = np.zeros_like(lmbda)
    min_r2 = np.zeros_like(lmbda)
    min_degrees = np.zeros_like(lmbda)

    for i in range(len(lmbda)):

        print('             lambda:',lmbda[i])  

        error_test, error_train, r2_test, *junk = bias_variance_tradeoff(x, y, z_noise, z_true, degree_max, lmbda=lmbda[i], return_var = True, cv_method=cv_method, reg_method=Ridge)#, normalize=normalize)

        min_r2[i] = np.max(r2_test)
        min_error[i] = np.min(error_test)
        print(np.argmin(error_test))
        min_degrees[i] = polydegree[np.argmin(error_test)]

        plt.plot(polydegree, error_test, label=(r'$\lambda$=%.2e' %lmbda[i]))

    #plt.plot(polydegree, error_train, label=(r'$\lambda=$%.2e Train Error', lmbda[i]))    
    idx_min = np.argmin(min_error)

    print('min lmbda:', lmbda[idx_min])
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

def lasso_bias_variance(x, y, z_noise, z_true, degree_max=14, lmbda_min=-6, lmbda_max=2, n_lmbda=20, plot_mse=False, normalize=False):

    lmbda = np.logspace(lmbda_min,lmbda_max,n_lmbda)
    #lmbda[-1] = 0
    N_trials = 1
 
    polydegree = np.arange(1,degree_max,1)
    min_error = np.zeros_like(lmbda)
    min_r2 = np.zeros_like(lmbda)
    min_degrees = np.zeros_like(lmbda)

    for i in range(len(lmbda)):

        print('             lambda:',lmbda[i])  

        error_test, error_train, r2_test, *junk = bias_variance_tradeoff(x, y, z_noise, z_true, degree_max, lmbda=lmbda[i], return_var = True, cv_method = cvmet.kfold_CV_sklearn, reg_method = Lasso, normalize=normalize)

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
        plt.plot(min_degrees[idx_min], min_error[idx_min], 'x')
        plt.xlabel('polynomial degree', size=14)
        plt.ylabel('Mean squared error', size=14)
        plt.tick_params(axis='both', labelsize=12)
        plt.grid('True', linestyle='dashed')
        plt.tight_layout()
        plt.legend()
        plt.show()

if __name__ == "__main__":
    main(OLSreg=True, bv_trade=True)
    #main_terrain(plot_beta=True, OLSreg=True)