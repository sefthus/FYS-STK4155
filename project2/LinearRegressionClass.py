import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import tools
import CV_methods as cv

class LinearRegressor:
    def __init__(self,
                X,
                z_true,
                z_noise,
                z_stddev,
                cv_method='sklearn'):

        self.X = X
        self.z_true = z_true
        self.z_noise = z_noise
        self.z_stddev = z_stddev

        self.cv_method = cv_method
        self.cv_methods()



    def cv_methods(self):
        cv_options ={
            'sklearn': cv.kfold_CV_sklearn,
            'numpy': cv.kfold_CV
        }
        self.cv_f = cv_options[self.cv_method]

    def OLSRegressor(self):
        z_pred_test, z_pred_train, mse_test, mse_train, r2_test = self.cv_f(self.X, self.z_noise, self.z_true, 
                                                                            stddev=self.z_stddev, reg_method=LinearRegression)

        self.z_pred_test = z_pred_test
        self.z_pred_train = z_pred_train
        self.mse_test = mse_test
        self.mse_train = mse_train
        self.r2_test = r2_test

    def RidgeRegressor(self, lmbda):
        z_pred_test, z_pred_train, mse_test, mse_train, r2_test = self.cv_f(self.X, self.z_noise, self.z_true, 
                                                                            stddev=self.z_stddev, lmbda=lmbda, reg_method=Ridge)
        self.z_pred_test = z_pred_test
        self.z_pred_train = z_pred_train
        self.mse_test = mse_test
        self.mse_train = mse_train
        self.r2_test = r2_test
        
    def LassoRegressor(self, lmbda):
        z_pred_test, z_pred_train, mse_test, mse_train, r2_test = cv.kfold_CV_sklearn(self.X, self.z_noise, 
                                                                            self.z_true, stddev=self.z_stddev, lmbda=lmbda, reg_method=Lasso)
        self.z_pred_test = z_pred_test
        self.z_pred_train = z_pred_train
        self.mse_test = mse_test
        self.mse_train = mse_train
        self.r2_test = r2_test
        
