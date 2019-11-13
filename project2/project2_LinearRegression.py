
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy import sparse

import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.neural_network import MLPRegressor


from original_NNclass import NeuralNetwork as oNeuralNetwork
from NNclass_arr import NeuralNetwork as aNeuralNetwork

import tools
from LinearRegressionClass import LinearRegressor as LinReg

seed = 3155
np.random.seed(seed)


def FrankeFunction(x,y):
    """ 
    calculates the Franke function from x and y from meshgrid
    """

    term1 = 0.75 * np.exp(-(0.25 * (9*x-2)**2) - 0.25 * ((9*y-2)**2))
    term2 = 0.75 * np.exp(-((9*x+1)**2) / 49.0 - 0.1 * (9*y+1))
    term3 = 0.5 * np.exp(-(9*x-7)**2 / 4.0 - 0.25 * ((9*y-3)**2))
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def linear_regression(x1, y1, z1_noise, z1_true, d=5, OLSreg=True, 
                        Ridgereg=True, Lassoreg=True, lmbd_R = 1e-2, 
                        lmbd_L=1e-4):

    if OLSreg: # Use linear regression
        X = tools.create_design_matrix(x1, y1, d=d)        
        LR = LinReg(X, z1_true, z1_noise, cv_method='numpy')

        print('   -------------OLS  cross validating')
        LR.OLSRegressor()
        print('mse test :', LR.mse_test)
        print('mse train:', LR.mse_train)
        print('r2 score :', LR.r2_test)

    # ------------- Ridge
    if Ridgereg:
        lmbd = lmbd_R
        X = tools.create_design_matrix(x1, y1, d=d)    
        LR = LinReg(X, z1_true, z1_noise, cv_method='numpy')

        print('   -------------Ridge regression cross validating')
        LR.RidgeRegressor(lmbd)# same result when using same random_state=0 in Kfold and shuffle
        print('mse test :', LR.mse_test)
        print('mse train:', LR.mse_train)
        print('r2 score :', LR.r2_test)
        
    # ------------- Lasso
    if Lassoreg:
        lmbd = 1e-4
        X = tools.create_design_matrix(x1, y1, d=d)    

        LR = LinReg(X, z1_true, z1_noise, cv_method='sklearn')

        print('   -------------Lasso regression cross validating')
        LR.LassoRegressor(lmbd)
        print('mse test :', LR.mse_test)
        print('mse train:', LR.mse_train)
        print('r2 score :', LR.r2_test)

def create_Franke(d=5, xy_datapoints=100, z_stddev=1, Franke_plot=False):
    
    n = 1./xy_datapoints # 0.01
    x = np.arange(0, 1, n)
    y = np.arange(0, 1, n)

    x, y = np.meshgrid(x, y)
    z = tools.FrankeFunction(x, y)

    if Franke_plot:
        Franke_plot(x,y,z)

    print(' turn data matrices to arrays')
    x1 = np.ravel(x)
    y1 = np.ravel(y)
    n = int(len(x1))

    z1_true = np.ravel(z)
    z1_noise = np.ravel(z) + np.random.normal(0, z_stddev, size=z1_true.shape) # adding noise

    print(' make design matrix')
    X = tools.create_design_matrix(x1, y1, d)

    return X, z1_noise, z1_true, x1, y1

def main_lingreg(calc_intercept=False, Franke_plot=False,turn_dense=False, do_linreg=True, OLSreg=True, Ridgereg=False, Lassoreg=False, do_NN=True, do_NN_np=False, do_NN_sk=True, vizualize_scores=True):
    
    X, z1_noise, z1_true, x1, y1 = create_Franke()

    # ---------- calculate y_pred using linear regression
    if do_linreg:
        linear_regression(x1, y1, z1_noise, z1_true, d=5, OLSreg=OLSreg, 
                            Ridgereg=Ridgereg, Lassoreg=Lassoreg, lmbd_R = 1e-2, 
                            lmbd_L=1e-6)

    # ------------ neural network
    if do_NN:
        print('\n ------- Neural network regession --------')

        training_share = 0.7
        X_train, X_test, z_train, z_test, z_train_true, z_test_true = train_test_split(X, z1_noise, z1_true, test_size=1-training_share, shuffle=True, random_state=seed)

        X_train, X_test, z_train, z_test, zscale_mean = tools.scale_data(X_train, X_test, z_train.reshape(-1,1), z_test.reshape(-1,1), cat_split=None)
        
        n_categories = 1 #=1 when lin. reg
        n_hidden_layers = 2

        n_hidden_neurons1 = int(np.floor(np.mean(n_categories + X_train.shape[1])))
        print('No of hidden neurons:',n_hidden_neurons1)
        
        n_hidden_neurons2 = int(np.round(X_train.shape[0]/(8*(n_categories+X_train.shape[1]))))
        print('No of hidden neurons:',n_hidden_neurons2)
        
        n_hidden_neurons3 = int(np.round(2./3*X_train.shape[1]) + n_categories)
        print('No of hidden neurons:',n_hidden_neurons3)
        #sys.exit()

        #hidden_layer_sizes = n_hidden_layers*[n_hidden_neurons,]
        #hidden_layer_sizes = [n_hidden_neurons, int(np.round(n_hidden_neurons/2))]
        print('no of input neurons', X_train.shape[1])


        if do_NN_np:
            epochs = 300
            batch_size = 40
            print(X_train.shape[0])
            #sys.exit()
            eta = 0.001
            lmbd = 0.0001

            n_hidden_layers = 1
            n_hidden_neurons = n_hidden_neurons1
            hidden_sizes = n_hidden_layers*[n_hidden_neurons,]
            #hidden_layer_sizes = [n_hidden_neurons, int(np.round(n_hidden_neurons/2))]
            print('No of hidden neurons:', hidden_sizes)
            
            init_method = 'Xavier'
            out_activation = 'linear'
            hidden_activation = 'sigmoid'
            cost_f = 'mse'

            #eta_vals = np.array([eta])
            #lmbd_vals = np.array([lmbd])
            eta_vals = np.logspace(-6, -1, 6)
            lmbd_vals = np.logspace(-6, -1, 6)

            # store the models for later use
            train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
            train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
            DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
            
            for i, eta in enumerate(eta_vals):
                costs = np.zeros((len(lmbd_vals),epochs))
                for j, lmbd in enumerate(lmbd_vals):
                    print('eta:',eta, 'lambda:', lmbd)
                    dnn = aNeuralNetwork(X_train, z_train, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                                        n_hidden_neurons=hidden_layer_sizes, n_categories=n_categories, init_method=init_method,
                                        out_activation = out_activation, hidden_activation=hidden_activation, cost_f = cost_f)

                    #dnn = oNeuralNetwork(X_train, z_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    #                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories, init_method=init_method)

                    dnn.train()

                    DNN_numpy[i][j] = dnn
        
                    z_pred_NN = dnn.predict(X_test) + zscale_mean
                    z_pred_NN_train = dnn.predict(X_train) + zscale_mean

                    train_mse[i][j] = mean_squared_error(z_train_true, z_pred_NN_train)
                    test_mse[i][j] = mean_squared_error(z_test_true, z_pred_NN)
                    train_r2[i][j] = r2_score(z_train_true, z_pred_NN_train)
                    test_r2[i][j] = r2_score(z_test_true, z_pred_NN)

                    costs[j,:] = dnn.cost_epoch
            
            plt.plot(np.linspace(1, epochs+1, epochs), dnn.cost_epoch, label=('eta = ', lmbd_vals[-1]))

            
            plt.xlabel('epoch')
            plt.ylabel('cost function')
            plt.ylim([-1,1e2])
            plt.show()
            arg_min = np.unravel_index(test_mse.argmin(), test_mse.shape)
            print('arg_min:', arg_min)
            print('best param: eta:', eta_vals[arg_min[0]], 'lambda:',lmbd_vals[arg_min[1]])
            print('best mse test score:', test_mse[arg_min])
            print('best mse train score:', train_mse[arg_min])
            print('best r2 test score:', test_r2[arg_min])
            print('best r2 train score:', train_r2[arg_min])

            if (len(eta_vals)<2 and len(lmbd_vals)<2):
                
                #prob_z_NN = dnn.predict_probabilities(X_test)
                #prob_z_NN_train = dnn.predict_probabilities(X_train)

                print('eta:', eta, 'lambda:', lmbd)
                print('hidden layers:', n_hidden_layers)
                print("MSE score NN: ", mean_squared_error(z_test_true, z_pred_NN))
                print('r1 score NN:', r2_score(z_test_true, z_pred_NN))

            elif vizualize_scores: 
                tools.vizualize_scores(test_mse, title='', xticks=lmbd_vals, 
                                yticks=eta_vals, xlab=r'$\lambda$', ylab=r'$\eta$')


        if do_NN_sk:
            # --------- NN with scikit learn
            # store models for later use    
            batch_size = 300
            eta = 0.01
            lmbd = 1e-6
            
            eta_vals = np.array([eta])
            lmbd_vals = np.array([lmbd])
            eta_vals = np.logspace(-6, -1, 7)
            lmbd_vals = np.logspace(-7, 1, 9)
            epochs = 1000
            
            DNN_sk = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
 
            train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
            train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
            
            hidden_size01 = (n_hidden_neurons1)
            hidden_size02 = (n_hidden_neurons2)
            hidden_size03 = (n_hidden_neurons3)

            hidden_size1 = n_hidden_layers*(n_hidden_neurons1,) # (neurons,neurons,...)
            hidden_size2 = (n_hidden_neurons1, int(np.round(n_hidden_neurons1/2)))

            hidden_size3 = n_hidden_layers*(n_hidden_neurons2,) # (neurons,neurons,...)
            hidden_size4 = (n_hidden_neurons2, int(np.round(n_hidden_neurons2/2)))

            hidden_size5 = n_hidden_layers*(n_hidden_neurons3,) # (neurons,neurons,...)
            hidden_size6 = (n_hidden_neurons3, int(np.round(n_hidden_neurons3/2)))


            #layers = [hidden_size01, hidden_size02, hidden_size03, hidden_size1, hidden_size2, hidden_size3, hidden_size4, hidden_size5, hidden_size6]
            layers = [hidden_size01, hidden_size02, hidden_size03, hidden_size1, hidden_size3, hidden_size5]

            param_grid = [
            {
            'activation' : ['logistic', 'relu'],
            'solver' : ['sgd'],
            'hidden_layer_sizes': layers,
            'alpha': lmbd_vals,
            'learning_rate_init': eta_vals,
            'batch_size': [40, 100, 500, 800, 1000],
            'max_iter': [300, 1000, 2000]
            }
            ]

            clf = GridSearchCV(MLPRegressor(), param_grid,
                                    scoring=['neg_mean_squared_error','r2'], 
                                    refit='neg_mean_squared_error')
            clf.fit(X_train, z_train.ravel())
            clf.predict(X_test)

            
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            sys.exit()
            '''
            for i, eta in enumerate(eta_vals):
                costs = np.zeros((len(lmbd_vals), epochs))
                for j, lmbd in enumerate(lmbd_vals):
                    print('eta:',eta, 'lambda:', lmbd)
    
                    dnn_sk = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                                        alpha=lmbd, solver='sgd', #learning_rate='invscaling', power_t=1,
                                        learning_rate_init=eta, max_iter=epochs, batch_size=batch_size)
                    dnn_sk.fit(X_train, z_train.ravel())
            
                    DNN_sk[i][j] = dnn_sk
        
                    z_pred_NN_sk = dnn_sk.predict(X_test) + zscale_mean
                    z_pred_NN_sk_train = dnn_sk.predict(X_train) + zscale_mean

                    train_mse[i][j] = mean_squared_error(z_train_true, z_pred_NN_sk_train)
                    test_mse[i][j] = mean_squared_error(z_test_true, z_pred_NN_sk)
                    train_r2[i][j] = r2_score(z_train_true, z_pred_NN_sk_train)
                    test_r2[i][j] = r2_score(z_test_true, z_pred_NN_sk)

                    #costs[j,:] = dnn_sk.loss_
                    print(dnn_sk.loss_)
            #plt.plot(np.linspace(1, epochs+1, epochs), dnn_sk.loss_cruve_, label=('eta = ', lmbd_vals[-1]))

            #plt.xlabel('epoch')
            #plt.ylabel('cost function')
            #plt.ylim([-1,1e2])
            #plt.show()
            arg_min = np.unravel_index(test_mse.argmin(), test_mse.shape)
            print('arg_max:', arg_min)
            print('best param: eta:', eta_vals[arg_min[0]], 'lambda:',lmbd_vals[arg_min[1]])
            print('best mse test score:', test_mse[arg_min])
            print('best mse train score:', train_mse[arg_min])
            print('best r2 test score:', test_r2[arg_min])
            print('best r2 train score:', train_r2[arg_min])
            '''
            if (len(eta_vals)<2 and len(lmbd_vals)<2):

                print('eta:', eta, 'lambda:', lmbd)
                print('hidden layers:', n_hidden_layers)
                print("mse NN: ", mean_squared_error(z_test_true, z_pred_NN_sk))
                print('r2 score NN:', r2_score(z_test_true, z_pred_NN_sk))

            elif vizualize_scores:
                tools.vizualize_scores(test_mse, title='', xticks=lmbd_vals, 
                                yticks=eta_vals, xlab=r'$\lambda$', ylab=r'$\eta$')
      




main_lingreg(Franke_plot=False, turn_dense=False, 
            do_linreg=False, OLSreg=True, Ridgereg=True, Lassoreg=True, 
            do_NN=True, do_NN_np=False, do_NN_sk=True, vizualize_scores=False)