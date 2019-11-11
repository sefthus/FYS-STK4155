
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import pandas as pd
from scipy import sparse
from scipy.special import expit
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, mean_squared_error, r2_score, f1_score
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier, MLPRegressor

from scikitplot.metrics import plot_cumulative_gain
from scikitplot.helpers import cumulative_gain_curve

from NNclass import NeuralNetwork
from original_NNclass import NeuralNetwork as oNeuralNetwork
from NNclass_arr import NeuralNetwork as aNeuralNetwork

import tools
from LinearRegressionClass import LinearRegressor as LinReg

seed = 3155
np.random.seed(seed)

# reading file into data frame

def create_df(filename=r'.\data\default of credit card clients.xls', remove_nan=True):
    """
        Creates a dataframe from filename using pandas. Removes invalid values
        from the factors. 
        Splits the dataframe into the design matrix X consisting of the factors,
        and the respone y.
        Onehotencodes the categorical variables in the design matrix X.
        Returns the design matrix X and respone y.

        filename  : the filename of the dataframe
        remove_nan: if True, removes invalid values from the factors
    """

    filename = filename
    nanDict = {}

    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month":"defaultPaymentNextMonth"}, inplace=True)
    #print(df)

    # Remove instances with zeros only for past bill statements or paid amounts
    # and not or, remove only when true in all columns
    df = df.drop(df[(df.BILL_AMT1 == 0) &
                    (df.BILL_AMT2 == 0) &
                    (df.BILL_AMT3 == 0) &
                    (df.BILL_AMT4 == 0) &
                    (df.BILL_AMT5 == 0) &
                    (df.BILL_AMT6 == 0)].index, axis=0)
                
    df = df.drop(df[(df.PAY_AMT1 == 0) &
                    (df.PAY_AMT2 == 0) &
                    (df.PAY_AMT3 == 0) &
                    (df.PAY_AMT4 == 0) &
                    (df.PAY_AMT5 == 0) &
                    (df.PAY_AMT6 == 0)].index, axis=0)
    #print(df)


    
    if remove_nan: # remove unspecified variables
        print('df shape before nan remove:',df.shape)
        print('df after removing nan:')
        # Instead of remove, keep in own NAN category, or add to other category?
        df = pay_remove_value(df,0)
        print('  remove pay=0:',df.shape)

        df = pay_remove_value(df,-2)
        print('  remove pay=-2', df.shape)

        df = edu_marr_remove_value(df)
        print('  remove edy=5,6, marriage=0:', df.shape)    



    # features and targets
    X = df.loc[:, df.columns !='defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns =='defaultPaymentNextMonth'].values

    # categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories='auto')
    #print(df.iloc[0:, 3])
    
    #print(X[0,:])
    # transform cat. var. columns into cat. variables.
    # new  columns are added at the start, columns before col 1 put behind new columns
    
    X = ColumnTransformer(
        [("",onehotencoder, [1,2,3, 5,6,7,8,9,10]),],
        remainder='passthrough'
        ).fit_transform(X)
    #print(len(X[0,:]))
    #print(X[0,:])
    #print(X[0,-14])
    #print(X[0,60:])
    #sys.exit()
    
    return X, y

def pay_remove_value(dframe, value):
    dframe = tools.drop_value(dframe, dframe.PAY_0, value)
    dframe = tools.drop_value(dframe, dframe.PAY_2, value)
    dframe = tools.drop_value(dframe, dframe.PAY_3, value)
    dframe = tools.drop_value(dframe, dframe.PAY_4, value)
    dframe = tools.drop_value(dframe, dframe.PAY_5, value)
    dframe = tools.drop_value(dframe, dframe.PAY_6, value)
    return dframe

def edu_marr_remove_value(dframe):
    
    dframe = tools.drop_value(dframe, dframe.EDUCATION, 5)
    dframe = tools.drop_value(dframe, dframe.EDUCATION, 6)
    dframe = tools.drop_value(dframe, dframe.MARRIAGE, 0)
    return dframe

def FrankeFunction(x,y):
    """ 
    calculates the Franke function from x and y from meshgrid
    """

    term1 = 0.75 * np.exp(-(0.25 * (9*x-2)**2) - 0.25 * ((9*y-2)**2))
    term2 = 0.75 * np.exp(-((9*x+1)**2) / 49.0 - 0.1 * (9*y+1))
    term3 = 0.5 * np.exp(-(9*x-7)**2 / 4.0 - 0.25 * ((9*y-3)**2))
    term4 = -0.2 * np.exp(-(9*x-4)**2 - (9*y-7)**2)

    return term1 + term2 + term3 + term4

def stochastic_gd(X, y, M=40, n_epochs=200, plot_cost=False):
    """ 
        Calculates the parameters regression parameters beta using stochastic
        gradient descent. 
        Returns beta

        X:        design matrix
        y:        predictor
        M:        number of minibatches
        n_epochs: number of epochs
        plot_cost: if True, plot the epochs against the cost function
    """

    n =  X.shape[0] # datapoints
    m = int(n/M) # number of minibatches
    t0 = 1.
    t1 = 10.

    cost_epoch = np.zeros(n_epochs)
    beta = np.random.randn(X.shape[1], 1) # initial beta parameters

    eta_j = t0/t1 # initial learning rate

    for epoch in range(1, n_epochs+1):
        X, y = shuffle(X, y)
        for i in range(m):
            k = i#np.random.randint(m) # pick random kth minibatch
            
            Xk = X[k*M:(k+1)*M,:]
            yk = y[k*M:(k+1)*M]

            if i == m-1: 
                Xk = X[k*M:,:]
                yk = y[k*M:,:]

            # compute gradient  and cost log reg
            sigmoid = 1 / (1 + np.exp(-Xk.dot(beta)))
            sigmoid_min = 1 / (1 + np.exp(Xk.dot(beta))) # =1-sigmoid(x) = sigmoid(-x) 
            
            cost_epoch[epoch-1] += -np.sum( yk*np.log(sigmoid) + (1-yk)*np.log(sigmoid_min) )

            gradient = - Xk.T.dot(yk - sigmoid)

            # compute new beta
            t = epoch*m + i
            eta_j = t0/(t+t1) # adaptive learning rate

            beta = beta - eta_j*gradient
            
    if plot_cost:
        plt.plot(np.linspace(1, n_epochs+1, n_epochs), cost_epoch)
        plt.xlabel('epoch')
        plt.ylabel('cost function')
        plt.show()

    return beta

def logreg_sklearn(X, y, no_grid=False):
    # --------- Logistic regression
    logReg = LogisticRegression(max_iter=1e4)
    y = y.ravel()
    if no_grid:
        logReg.fit(X, y)

        return logReg

    else:
        lambdas = np.logspace(-5, 7, 13)
        #parameters = [{'C':1/lambdas, "solver":["lbfgs"]}]
        parameters = [{'C':1/lambdas, "solver":["sgd"]}]
        scoring =['accuracy', 'roc_auc']
        gridSearch = GridSearchCV(logReg, parameters,cv=5, scoring=scoring, refit='roc_auc')
        gridSearch.fit(X, y)

        return gridSearch

def log_reg(X_train, X_test, y_train, M=40, n_epochs=200, plot_cost=False):
    """
        Performs a logistic regression of a binary respinse, using 
        stochastic gradient descent.
        Returns the predicted outcome (y_pred_b), the probability of 
        outcome 0 (prob_y_b[:,0]) and 1 (prob_y_b[:,1]) for test and training data, in that order.

        X_train:   training data design matrix
        X_test:    test data
        y_train:   training response
        M:         points in each batch, to be used in the SGD
        n_epochs:  number of epochs used in SGD
        plot_cost: wether or not to plot the cost function, passed onto SGD solver

    """
    beta = stochastic_gd(X_train, y_train, M, n_epochs, plot_cost)# , M=len(X_train)) # =gradient descent

    # probabilities of non-default (0) and default (1)    
    prob_y_test = 1/(1 + np.exp(-X_test.dot(beta)))
    prob_y_test_b = np.column_stack([1-prob_y_test, prob_y_test])

    prob_y_train = 1/(1 + np.exp(-X_train.dot(beta)))
    prob_y_train_b = np.column_stack([1-prob_y_train, prob_y_train])

    # round up or down y_pred to get on binary form
    y_pred = np.zeros_like(prob_y_test)
    y_pred[prob_y_test >= 0.5] = 1
    y_pred[prob_y_test < 0.5] = 0

    return y_pred, prob_y_test_b, prob_y_train_b

def auc_CGC(y_true, prob_y):
    """ calculate area under cumulativ gain curve """

    if prob_y.shape[1]>1:
        prob_y = prob_y[:,1]

    x_data, y_data = cumulative_gain_curve(y_true, prob_y)
    x_best, y_best = tools.bestCurve(y_true)
    x_base, y_base = np.array([0,1]), np.array([0,1])
    #plt.plot(x_data,y_data)
    #plt.show()
    auc_data = np.trapz(y_data, x_data)#- np.trapz(y_best, x_best)
    auc_best = np.trapz(y_best, x_best)#- np.trapz(y_best, x_best)
    auc_base = np.trapz(y_base, x_base)
    area_ratio = (auc_data - auc_base) / (auc_best-auc_base)
    return auc_data, area_ratio

def plot_cum_gain_chart(y_true, prob_y, prob_ysk=None):

    lw = 2
    if prob_ysk is not None:
        x_data, y_data = cumulative_gain_curve(y_true, prob_y[:,1])
        x_data_sk, y_data_sk = cumulative_gain_curve(y_true, prob_ysk[:,1])

        plt.plot(x_data, y_data, linewidth=lw, label='model')
        plt.plot(x_data_sk, y_data_sk, '--', linewidth=lw, label='model sklearn')

    else:
        x_data, y_data = cumulative_gain_curve(y_true, prob_y[:,1])
        plt.plot(x_data, y_data, linewidth=lw, label='model')

    x_best, y_best = tools.bestCurve(y_true)
    x_base, y_base = np.array([0,1]), np.array([0,1])

    plt.plot(x_best, y_best, linewidth=lw, label='best curve')
    plt.plot(x_base, y_base, ':', linewidth=lw, label='baseline')
    plt.xlabel(r'Fraction of data', size=14)
    plt.ylabel(r'Cumulative gain', size=14)
    plt.legend(prop={'size': 12})
    plt.grid('True', linestyle='dashed')
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.show()


def main_logreg(remove_nan=True, turn_dense=False, do_logreg=True, do_NN=True, do_NN_np=False, do_NN_sk=True, plot_cumgain_test=False, plot_cumgain_train=False, vizualize_scores=True):
    
    X, y = create_df(remove_nan=remove_nan)

    training_share = 0.7
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-training_share, shuffle=True, random_state=seed)

    X_train, X_test = tools.scale_data(X_train, X_test, turn_dense=turn_dense)
    
    # ---------- calculate y_pred using log. reg. and stochastic gradient descent
    if do_logreg:
        print('\n ---------- Log reg with SGD')
        M=40
        n_epochs = 300

        # probabilities of non-default (0) and default (1)    
        y_pred, prob_y_test, prob_y_train =  log_reg(X_train, X_test, y_train, M=M, n_epochs=n_epochs, plot_cost=False)
        
        auc_test, area_ratio_test = auc_CGC(y_test, prob_y_test)

        print('ytest sum:', np.sum(y_test))
        print('yhat sum:', np.sum(y_pred)) # ensure not only predicting zero's
        null_score = np.max(y_test.mean(), 1 - y_test.mean())
        print('Accuracy predicting majority class:', null_score)
        print('accuracy score:', accuracy_score(y_test, y_pred))
        print('f1 score:', f1_score(y_test, y_pred))
        print('AUC CGC:', auc_test)
        print('Area rato:', area_ratio_test)

        # -------- calcuate y_pred using sklearn's logistic regression
        print('\n ---------- Log reg with sklearn')
        model = logreg_sklearn(X_train, y_train)
        y_pred_sk = model.predict(X_test)

        # probabilities of non-default 0 and default 1
        prob_ysk_test = model.predict_proba(X_test)
        prob_ysk_train = model.predict_proba(X_train)

        auc_sk_test, area_ratio_sk_test = auc_CGC(y_test, prob_ysk_test)

        print('yhat sklearn sum:', np.sum(y_pred_sk))
        null_score = np.max(y_test.mean(), 1 - y_test.mean())
        print('Accuracy predicting majority class:', null_score)
        print('accuracy score sklearn:', accuracy_score(y_test, y_pred_sk))
        print('f1 score sklearn:', f1_score(y_test, y_pred_sk))
        print('AUC CGC:', auc_sk_test)
        print('Area rato:', area_ratio_sk_test)



        if plot_cumgain_test:
            plot_cum_gain_chart(y_test, prob_y_test, prob_ysk_test)#, prob_ysk)
            print('prob shape:', prob_y_test.shape)
        
        if plot_cumgain_train:
            plot_cum_gain_chart(y_train, prob_y_train, prob_ysk_train)


    # ------------ neural network
    if do_NN:
        print('\n ------- Neural network regession --------')
        epochs = 50
        batch_size = 100
        eta = 1e-2
        lmbd = 1e-3
        n_hidden_layers = 1
        #n_hidden_neurons = int(np.round(np.mean(n_categories + X_train.shape[1])))
        #print('No of hidden neurons:',n_hidden_neurons)
        #n_hidden_neurons = int(np.round(X_train.shape[0]/(2*(n_categories+X_train.shape[1]))))
        #print('No of hidden neurons:',n_hidden_neurons)
        n_hidden_neurons = int(np.round(2./3*X_train.shape[1]) + n_categories)
        #print('No of hidden neurons:',n_hidden_neurons)
        #sys.exit()
        hidden_layer_sizes = n_hidden_layers*[n_hidden_neurons,]
        #hidden_layer_sizes = [n_hidden_neurons, int(np.round(n_hidden_neurons/2))]
        print('No of hidden neurons:', hidden_layer_sizes)

        if do_NN_np:
            epochs = 50
            batch_size = 100
            n_categories = 2 #=1 when lin. reg
            
            init_method = 'Xavier'
            out_activation = 'softmax'
            hidden_activation = 'sigmoid'
            cost_f = 'ce'

            y_train_onehot = OneHotEncoder(categories='auto').fit_transform(y_train).toarray()
            

            #eta_vals = np.array([eta])
            #lmbd_vals = np.array([lmbd])

            eta_vals = np.logspace(-7, -1, 7)
            lmbd_vals = np.logspace(-7, -1, 7)
            
            train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
            train_auc = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_auc = np.zeros((len(eta_vals), len(lmbd_vals)))

            # store the models for later use
            DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
            
            for i, eta in enumerate(eta_vals):
                for j, lmbd in enumerate(lmbd_vals):
                    #print('eta:',eta, 'lambda:', lmbd)
                    dnn = aNeuralNetwork(X_train, y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                                        n_hidden_neurons=hidden_layer_sizes, n_categories=n_categories, init_method=init_method,
                                        out_activation = out_activation, hidden_activation=hidden_activation, cost_f = cost_f)

                    #dnn = oNeuralNetwork(X_train, y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                    #                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories, init_method=init_method)

                    dnn.train()

                    DNN_numpy[i][j] = dnn
        
                    y_pred_NN = dnn.predict(X_test)
                    y_pred_NN_train = dnn.predict(X_train)
                    prob_y_NN = dnn.predict_probabilities(X_test)
                    prob_y_NN_train = dnn.predict_probabilities(X_train)

                    train_accuracy[i][j] = accuracy_score(y_train, y_pred_NN_train)
                    test_accuracy[i][j] = accuracy_score(y_test,y_pred_NN)
                    auc_train, area_ratio_train = auc_CGC(y_train, prob_y_NN_train)
                    train_auc[i][j] = area_ratio_train
                    #train_auc[i][j]= roc_auc_score(y_train,y_pred_NN_train)
                    auc_test, area_ratio_test= auc_CGC(y_test, prob_y_NN)
                    test_auc[i][j] = area_ratio_test
                    #test_auc[i][j] = roc_auc_score(y_test, y_pred_NN)

            arg_max = np.unravel_index(test_auc.argmax(), test_auc.shape)
            print('arg_max:', arg_max)
            print('best param: eta:', eta_vals[arg_max[0]], 'lambda:',lmbd_vals[arg_max[1]])
            print('best accuracy test score:', test_accuracy[arg_max])
            print('best accuracy train score:', train_accuracy[arg_max])

            print('best area ratio test score:', test_auc[arg_max])
            print('best area ratio score:', train_auc[arg_max])

            if (len(eta_vals)<2 and len(lmbd_vals)<2):

                prob_y_NN = dnn.predict_probabilities(X_test)
                prob_y_NN_train = dnn.predict_probabilities(X_train)
                auc_NN_test, area_ratio_NN_test = auc_CGC(y_test, prob_y_NN)

                print('\n Neural Network classification')
                print('ytest sum:', y_test.sum())
                print('hidden layers:', n_hidden_layers)
                print('yhat NN sum:', y_pred_NN.sum())
                print('confusion test :', confusion_matrix(y_test, y_pred_NN))
                print('confusion train:', confusion_matrix(y_train, y_pred_NN_train))    
                # accuracy score from scikit library
                null_score = np.max(y_test.mean(), 1 - y_test.mean())
                print('Accuracy predicting majority class:', null_score)
                print("Accuracy score NN: ", accuracy_score(y_test, y_pred_NN))
                print('f1 score NN:', f1_score(y_test, y_pred_NN))
                print('AUC CGC:', auc_NN_test)
                print('Area ratio:', area_ratio_NN_test)

                if plot_cumgain_test:
                    #print('prob y:', prob_y_NN.shape)
                    plot_cum_gain_chart(y_test, prob_y_NN)
                if plot_cumgain_train:
                    #print('prob y:', prob_y_NN.shape)
                    plot_cum_gain_chart(y_train, prob_y_NN_train)
                
            elif vizualize_scores:

                tools.vizualize_scores(test_auc, title='', xticks=lmbd_vals, 
                                    yticks=eta_vals, xlab=r'$\lambda$', ylab=r'$\eta$')

        if do_NN_sk:
            # --------- NN with scikit learn
            print('\n ---------- Neural Network classification sklearn')
            
            batch_size = 100
            epochs=1000
            #eta_vals = np.array([eta])
            #lmbd_vals = np.array([lmbd])

            eta_vals = np.logspace(-7, 0, 8)
            lmbd_vals = np.logspace(-7, 0, 8)
            
            # store models for later use
            DNN_sk = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
            
            
            train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
            train_auc = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_auc = np.zeros((len(eta_vals), len(lmbd_vals)))
            
            hidden_layer_sizes = n_hidden_layers*(n_hidden_neurons,) # (neurons,neurons,...)
            for i, eta in enumerate(eta_vals):
                for j, lmbd in enumerate(lmbd_vals):
                    #print('eta:',eta, 'lambda:', lmbd)
    
                    dnn_sk = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='relu',
                                        alpha=lmbd, solver='adam', learning_rate_init=eta, max_iter=epochs)
                    dnn_sk.fit(X_train, y_train.ravel())
                    #y_pred_NN_sk = dnn_sk.predict(X_test)
                    #prob_y_NN_sk = dnn_sk.predict_proba(X_test)
            
            
                    DNN_sk[i][j] = dnn_sk
        
                    y_pred_NN_sk = dnn_sk.predict(X_test)
                    y_pred_NN_sk_train = dnn_sk.predict(X_train)

                    train_accuracy[i][j] = accuracy_score(y_train, y_pred_NN_sk_train)
                    test_accuracy[i][j] = accuracy_score(y_test,y_pred_NN_sk)
                    train_auc[i][j] = roc_auc_score(y_train,y_pred_NN_sk_train)
                    test_auc[i][j] = roc_auc_score(y_test,y_pred_NN_sk)

            arg_max = np.unravel_index(test_auc.argmax(), test_auc.shape)
            print('arg_max:', arg_max)
            print('best param: eta:', eta_vals[arg_max[0]], 'lambda:',lmbd_vals[arg_max[1]])
            print('best AUC test score:', test_auc[arg_max])
            print('best AUC train score:', train_auc[arg_max])

            if (len(eta_vals)< 2 and len(lmbd_vals<2)):

                prob_y_NN_sk = dnn_sk.predict_proba(X_test)
                prob_y_NN_sk_train = dnn_sk.predict_proba(X_train)
                auc_NN_sk_test, area_ratio_NN_sk_test = auc_CGC(y_test, prob_y_NN_sk)


                print('ytest sum:', y_test.sum())
                print('hidden layers:', n_hidden_layers)
                print('yhat NN sum:', y_pred_NN_sk.sum())
                print('confusion test :', confusion_matrix(y_test, y_pred_NN_sk))
                print('confusion train:', confusion_matrix(y_train, y_pred_NN_sk_train))
                # accuracy score from scikit library
                print("Accuracy score NN: ", accuracy_score(y_test, y_pred_NN_sk))
                print('f1 score NN:', f1_score(y_test, y_pred_NN_sk))
                print('AUC CGC:', auc_NN_sk_test)
                print('Area rato:', area_ratio_NN_sk_test)

                if plot_cumgain_test:
                    #print('prob y:', prob_y_NN.shape)
                    plot_cum_gain_chart(y_test, prob_y_NN, prob_y_NN_sk)
                if plot_cumgain_train:
                    #print('prob y:', prob_y_NN.shape)
                    plot_cum_gain_chart(y_train, prob_y_NN_sk_train)

            elif vizualize_scores:
                tools.vizualize_scores(test_auc, title='', xticks=lmbd_vals, 
                                    yticks=eta_vals, xlab=r'$\lambda$', ylab=r'$\eta$')
                '''
                #print("Accuracy score on test set: ", dnn_sk.score(X_test, y_test.ravel()))
                print('yhat NN sum sklearn:', y_pred_NN_sk.sum())
                print("Accuracy score NN sklearn: ", accuracy_score(y_test, y_pred_NN_sk))
                print('f1 score NN sklearn:', f1_score(y_test, y_pred_NN_sk))
                #sys.exit()
                # Things are looking good! same results sk and mine
                '''


def main_lingreg(calc_intercept=False, Franke_plot=False,turn_dense=False, do_linreg=True, OLSreg=True, Ridgereg=False, Lassoreg=False, do_NN=True, do_NN_np=False, do_NN_sk=True, vizualize_scores=True):
    
    # Make data.
    print(' make data')
    x = np.arange(0, 1, 0.01)
    y = np.arange(0, 1, 0.01)

    x, y = np.meshgrid(x, y)
    z = tools.FrankeFunction(x, y)

    if Franke_plot:
        Franke_plot(x,y,z)

    print(' turn data matrices to arrays')
    x1 = np.ravel(x)
    y1 = np.ravel(y)
    n = int(len(x1))

    z1_true = np.ravel(z)
    stddev = 1
    z1_noise = np.ravel(z) + np.random.normal(0, stddev, size=z1_true.shape) # adding noise

    print(' make design matrix')
    #X = tools.create_design_matrix(x, y, d=5)
    #print(z1_noise.shape, X.shape)
    #sys.exit()

    # ---------- calculate y_pred using linear regression
    if do_linreg:

        if OLSreg: # Use linear regression
            d=5
            X = tools.create_design_matrix(x1, y1, d=d)        
            LR = LinReg(X, z1_true, z1_noise, stddev, cv_method='numpy')

            print('   -------------OLS  cross validating')
            LR.OLSRegressor()# same result when using same random_state=0 in Kfold and shuffle
            print('mse test :', LR.mse_test)
            print('mse train:', LR.mse_train)
            print('r2 score :', LR.r2_test)

        # ------------- Ridge
        if Ridgereg:
            d=5
            lmbd = 1e-2
            X = tools.create_design_matrix(x1, y1, d=d)    
            LR = LinReg(X, z1_true, z1_noise, stddev, cv_method='numpy')

            print('   -------------Ridge regression cross validating')
            LR.RidgeRegressor(lmbd)# same result when using same random_state=0 in Kfold and shuffle
            print('mse test :', LR.mse_test)
            print('mse train:', LR.mse_train)
            print('r2 score :', LR.r2_test)
            
        # ------------- Lasso
        if Lassoreg:
            d=5
            lmbd = 1e-4
            X = tools.create_design_matrix(x1, y1, d=d)    

            LR = LinReg(X, z1_true, z1_noise, stddev, cv_method='numpy')

            print('   -------------Lasso regression cross validating')
            LR.LassoRegressor(lmbd)# same result when using same random_state=0 in Kfold and shuffle
            print('mse test :', LR.mse_test)
            print('mse train:', LR.mse_train)
            print('r2 score :', LR.r2_test)

        if plot_cumgain_test:
            plot_cum_gain_chart(LR.z_test, LR.prob_z)
        
        if plot_cumgain_train:
            plot_cum_gain_chart(LR.z_train, LR.prob_z_train)

    # ------------ neural network
    if do_NN:
        print('\n ------- Neural network regession --------')

        d=5
        X = tools.create_design_matrix(x1, y1, d=d)
        training_share = 0.7
        X_train, X_test, z_train, z_test, z_train_true, z_test_true = train_test_split(X, z1_noise, z1_true, test_size=1-training_share, shuffle=True, random_state=seed)

        X_train, X_test, z_train, z_test, zscale_mean = tools.scale_data(X_train, X_test, z_train.reshape(-1,1), z_test.reshape(-1,1), cat_split=None)
        
        n_categories = 1 #=1 when lin. reg
        n_hidden_layers = 1
        n_hidden_neurons = int(np.round(2./3*X_train.shape[1]) + n_categories)
        
        #n_hidden_neurons = int(np.floor(np.mean(n_categories + X_train.shape[1])))
        #print('No of hidden neurons:',n_hidden_neurons)
        
        #n_hidden_neurons = int(np.round(X_train.shape[0]/(5*(n_categories+X_train.shape[1]))))
        #print('No of hidden neurons:',n_hidden_neurons)
        
        #n_hidden_neurons = int(np.round(2./3*X_train.shape[1]) + n_categories)
        #print('No of hidden neurons:',n_hidden_neurons)
        #sys.exit()

        hidden_layer_sizes = n_hidden_layers*[n_hidden_neurons,]
        #hidden_layer_sizes = [n_hidden_neurons, int(np.round(n_hidden_neurons/2))]
        print('no of input neurons', X_train.shape[1])
        print('No of hidden neurons:', hidden_layer_sizes)


        if do_NN_np:
            epochs = 1000
            batch_size = 200
            print(X_train.shape[0])
            #sys.exit()
            eta = 0.001
            lmbd = 0.1

            init_method = 'Xavier'
            out_activation = 'linear'
            hidden_activation = 'sigmoid'
            cost_f = 'mse'

            #eta_vals = np.array([eta])
            eta_vals = np.logspace(-6, -2, 5)
            #lmbd_vals = np.array([lmbd])
            lmbd_vals = np.logspace(-7, -1, 7)

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
            print('arg_max:', arg_min)
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
                print("MSE score NN: ", mean_squared_error(z_test, z_pred_NN))
                print('r1 score NN:', r2_score(z_test, z_pred_NN))

            elif vizualize_scores: 
                tools.vizualize_scores(test_mse, title='', xticks=lmbd_vals, 
                                yticks=eta_vals, xlab=r'$\lambda$', ylab=r'$\eta$')


        if do_NN_sk:
            # --------- NN with scikit learn
            # store models for later use    
            batch_size = 600
            eta = 0.001
            lmbd = 0.01
            
            #eta_vals = np.array([eta])
            lmbd_vals = np.array([lmbd])
            eta_vals = np.logspace(-6, -2, 5)
            lmbd_vals = np.logspace(-7, -1, 7)
            epochs = 1000
            
            DNN_sk = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
 
            train_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_mse = np.zeros((len(eta_vals), len(lmbd_vals)))
            train_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_r2 = np.zeros((len(eta_vals), len(lmbd_vals)))
            
            hidden_layer_sizes = n_hidden_layers*(n_hidden_neurons,) # (neurons,neurons,...)
            #hidden_layer_sizes = (n_hidden_neurons, int(np.round(n_hidden_neurons/2)))
            
            for i, eta in enumerate(eta_vals):
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


            arg_min = np.unravel_index(test_mse.argmin(), test_mse.shape)
            print('arg_max:', arg_min)
            print('best param: eta:', eta_vals[arg_min[0]], 'lambda:',lmbd_vals[arg_min[1]])
            print('best mse test score:', test_mse[arg_min])
            print('best mse train score:', train_mse[arg_min])
            print('best r2 test score:', test_r2[arg_min])
            print('best r2 train score:', train_r2[arg_min])
            
            if (len(eta_vals)<2 and len(lmbd_vals)<2):

                print('eta:', eta, 'lambda:', lmbd)
                print('hidden layers:', n_hidden_layers)
                print("mse NN: ", mean_squared_error(z_test, z_pred_NN_sk))
                print('r2 score NN:', r2_score(z_test, z_pred_NN_sk))

            elif vizualize_scores:
                tools.vizualize_scores(test_mse, title='', xticks=lmbd_vals, 
                                yticks=eta_vals, xlab=r'$\lambda$', ylab=r'$\eta$')
            

#main_logreg(remove_nan=True, turn_dense=False, do_logreg=False, do_NN=True, do_NN_np=False, do_NN_sk=True, plot_cumgain_test=True, vizualize_scores=False)

main_lingreg(Franke_plot=False, turn_dense=False, 
            do_linreg=False, OLSreg=True, Ridgereg=True, Lassoreg=True, 
            do_NN=True, do_NN_np=False, do_NN_sk=True, vizualize_scores=True)