
from mpl_toolkits.mplot3d import Axes3D
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

def best_fit(X, z_noise, z_true, stddev=1, lmbda=0, normalize=False, center_var = True, include_intercept=False, test_centering = False, use_sklearn=False, reg_method=Ridge):
    """ do a regression analysis without cross validating      """
    """ choose to mean-center variables and test the centering """
    """ chose to use sklearn or not                            """
    #X = create_design_matrix(x, y, d) 

    if center_var: # mean-center X and z

        if tools.check_eq(X[:,0], 1) and not include_intercept:
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
            XtXinv = tools.invert_matrix(np.matmul(X_c.T, X_c) + lmbda*Idm)

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
            if include_intercept and not tools.check_eq(X[:,0], 1):
                beta = np.append(np.array([scalerz.mean_]), beta.ravel())

        else:

            X_c = (X - np.mean(X, axis=0))/np.std(X, axis=0)

            Idm = np.identity(np.shape(X_c.T)[0])
            XtXinv = toold.invert_matrix(np.matmul(X_c.T, X_c) + lmbda*Idm)

            z_noise_mean = np.mean(z_noise)
            z_noise_c = z_noise - z_noise_mean

            #z_true_mean = np.mean(z_true)
            z_true_c = z_true# - z_true_mean

            beta = XtXinv.dot(X_c.T).dot(z_noise_c)
            z_tilde = (np.matmul(X_c, beta) + z_noise_mean)
            beta = beta#np.append(z_noise_mean, beta.ravel())
            if include_intercept and not toold.check_eq(X[:,0], 1):
                beta = np.append(z_noise_mean, beta.ravel())

        if test_centering:
            tools.check_centering(X, X_c)
            tools.check_centering(z_true, z_true_c)
            tools.check_centering(z_noise, z_noise_c)
        
        varbeta = stddev**2*np.diag(XtXinv)
        if include_intercept and not check_eq(X[:,0], 1):
            varbeta = np.append(stddev**2*np.var(z_noise)/len(z_noise),varbeta)
    else:

        Idm = np.identity(np.shape(X.T)[0])
        XtXinv = tools.invert_matrix(np.matmul(X.T,X) + lmbda*Idm)
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
