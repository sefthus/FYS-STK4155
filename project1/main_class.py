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

import tools

class CrossValidation:

    def __init__(selfs)
    self.x = np.random.random(50)
    #self.x = np.arange(0, 1, 0.01)
    self.y = np.random.random(50)
    #self.y = np.arange(0, 1, 0.01)
    self.x, self.y = np.meshgrid(x,y)

    self.z = self.frankefunction

    self.x1 = np.ravel(self.x)
    self.y1 = np.ravel(self.y)

def FrankeFunction(self, x,y):
    """ the Franke function """

    term1 = 0.75 * np.exp(-(0.25 * (9*x-2)**2) - 0.25 * ((9*y-2)**2))
    term2 = 0.75 * np.exp(-((9*x+1)**2) / 49.0 - 0.1 * (9*y+1))
    term3 = 0.5 * np.exp(-(9*x-7)**2 / 4.0 - 0.25 * ((9*y-3)**2))
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4