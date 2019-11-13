
#from mpl_toolkits.mplot3d import Axes3D
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
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,  OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

from scikitplot.metrics import plot_cumulative_gain
from scikitplot.helpers import cumulative_gain_curve

from original_NNclass import NeuralNetwork as oNeuralNetwork
from NNclass_arr import NeuralNetwork as aNeuralNetwork

import tools

from LogisticRegressionClass import LogisticRegressor

seed = 3155
np.random.seed(seed)


def create_df(filename=r'.\data\default of credit card clients.xls', remove_pay0=True, resample=False):
    """
        Creates a dataframe from filename using pandas. Removes invalid values
        from the factors. 
        Splits the dataframe into the design matrix X consisting of the factors,
        and the respone y.
        Onehotencodes the categorical variables in the design matrix X.
        Returns the design matrix X and respone y.

        filename  : the filename of the dataframe
        remove_pay0: if True, removes all instances where PAY_X is zero, this will
                     remove over 80% of the data
        resample: if True, SMOTE resampling is used to balance the data
    """

    filename = filename
    nanDict = {}

    df = pd.read_excel(filename, header=1, skiprows=0, index_col=0, na_values=nanDict)
    df.rename(index=str, columns={"default payment next month":"defaultPaymentNextMonth"}, inplace=True)

    # Remove instances with zeros only for past bill statements or paid amounts
    # and not or, remove only when true in all columns
    print('before removing instances where all bill statements or paid amount is zero:', df.shape)
    
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
    
    print('after removing instances where all bill statements or paid amount is zero:', df.shape)

    
    
    print('df shape before illegal values removed:',df.shape)
    print('df after removing illegals:')

    df = pay_remove_value(df,-2)
    print('  remove pay=-2', df.shape)

    df = bill_amt_remove_negative(df, 0)
    print('  remove Pay_amt, bill_amt <0:', df.shape)


    df = edu_marr_remove_value(df)
    print('  remove edy=0,5,6, marriage=0:', df.shape)

    if remove_pay0:# over 80 % of data lost

        df = pay_remove_value(df,0)
        print('  remove pay=0:',df.shape)



    # features and targets
    X = df.loc[:, df.columns !='defaultPaymentNextMonth'].values
    y = df.loc[:, df.columns =='defaultPaymentNextMonth'].values

    # categorical variables to one-hot's
    onehotencoder = OneHotEncoder(categories='auto')
    #print(df.iloc[0:, 3])
    
    # transform cat. var. columns into cat. variables.
    # new  columns are added at the start, columns before col 1 put behind new columns
    
    X = ColumnTransformer(
        [("",onehotencoder, [1,2,3, 5,6,7,8,9,10]),],
        remainder='passthrough'
        ).fit_transform(X)
    print(' shape of dataset without resampling', X.shape,y.shape)

    if resample:
        sm = SMOTE(random_state=seed)
        X, y = sm.fit_resample(X, y.ravel())
        y = y.reshape(-1,1)
        print(' shape of dataset after resampling', X.shape,y.shape)
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

def bill_amt_remove_negative(dframe, value):
    dframe = tools.drop_value(dframe, dframe.BILL_AMT1, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.BILL_AMT2, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.BILL_AMT3, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.BILL_AMT4, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.BILL_AMT5, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.BILL_AMT6, value, drop_less_than=True)

    dframe = tools.drop_value(dframe, dframe.PAY_AMT1, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.PAY_AMT2, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.PAY_AMT3, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.PAY_AMT4, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.PAY_AMT5, value, drop_less_than=True)
    dframe = tools.drop_value(dframe, dframe.PAY_AMT6, value, drop_less_than=True)

    return dframe

def edu_marr_remove_value(dframe):


    #sys.exit()
    dframe = tools.drop_value(dframe, dframe.EDUCATION, 0)
    dframe = tools.drop_value(dframe, dframe.EDUCATION, 5)
    dframe = tools.drop_value(dframe, dframe.EDUCATION, 6)
    dframe = tools.drop_value(dframe, dframe.MARRIAGE, 0)

    return dframe

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

def logistic_regression(X_train, X_test, y_train, y_test, 
                        M=40, n_epochs=300, plot_cost=False, save=False, plot_cumgain_test = False, 
                        plot_cumgain_train=False):

    # baseline accuracy score
    null_score = np.max([y_test.mean(), 1 - y_test.mean()])

    print('\n ---------- Log reg with SGD')
    # probabilities of non-default (0) and default (1)    
    LogReg = LogisticRegressor(X_train, y_train)
    LogReg.fit_numpy(M=M, n_epochs=n_epochs) # uses SGD
    y_pred_test = LogReg.predict_numpy(X_test)
    y_pred_train = LogReg.predict_numpy(X_train)
    prob_y_test = LogReg.predict_probabilities_numpy(X_test)
    prob_y_train = LogReg.predict_probabilities_numpy(X_train)

    if plot_cost:
        LogReg.plot_cost_function(save=save, name='cost_eta_decay_full')

    auc_test, area_ratio_test = auc_CGC(y_test, prob_y_test)

    print('ytest sum:', np.sum(y_test))
    print('yhat sum:', np.sum(y_pred_test)) # ensure not only predicting zero's

    print('y test mean', y_test.mean())
    print('yhat mean:', np.mean(y_pred_test)) # ensure not only predicting zero's

    print('accuracy baseline:', null_score)
    print('accuracy score test:', accuracy_score(y_test, y_pred_test))
    print('accuracy score train:', accuracy_score(y_train, y_pred_train))
    print('f1 score test:', f1_score(y_test, y_pred_test))
    print('f1 score train:', f1_score(y_train, y_pred_train))
    print('AUC CGC:', auc_test)
    print('Area ratio:', area_ratio_test)

    a=accuracy_score(y_test, y_pred_test)
    b=accuracy_score(y_train, y_pred_train)
    #print(a,b)
    #return a,b
    #sys.exit()
    # -------- calcuate y_pred using sklearn's logistic regression
    print('\n ---------- Log reg with sklearn')

    LogReg = LogisticRegressor(X_train, y_train)
    LogReg.fit_sklearn()
    y_pred_sk = LogReg.predict_sklearn(X_test)
    y_pred_sk_train = LogReg.predict_sklearn(X_train)

    # probabilities of non-default 0 and default 1
    prob_ysk_test = LogReg.predict_probabilities_sklearn(X_test)
    prob_ysk_train = LogReg.predict_probabilities_sklearn(X_train)

    auc_sk_test, area_ratio_sk_test = auc_CGC(y_test, prob_ysk_test)
    auc_sk_train, area_ratio_sk_train = auc_CGC(y_train, prob_ysk_train)

    print('yhat sklearn sum:', np.sum(y_pred_sk))

    print('accuracy baseline:', null_score)
    print('accuracy score sklearn test:', accuracy_score(y_test, y_pred_sk))
    print('accuracy score sklearn train:', accuracy_score(y_train, y_pred_sk_train))
    print('f1 score sklearn test:', f1_score(y_test, y_pred_sk))
    print('f1 score sklearn train:', f1_score(y_train, y_pred_sk_train))
    print('AUC CGC:', auc_sk_test)
    print('Area ratio:', area_ratio_sk_test)


    if plot_cumgain_test:
        plot_cum_gain_chart(y_test, prob_y_test, prob_ysk_test)
    
    if plot_cumgain_train:
        plot_cum_gain_chart(y_train, prob_y_train, prob_ysk_train)


def main_logreg(remove_pay0=True, resample=False, turn_dense=False, do_logreg=True, do_NN=True, do_NN_np=False, do_NN_sk=True, plot_cumgain_test=False, plot_cumgain_train=False, vizualize_scores=True):
    
    X, y = create_df(remove_pay0=remove_pay0, resample=resample)

    training_share = 0.7
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-training_share, shuffle=True, random_state=seed)

    X_train, X_test = tools.scale_data(X_train, X_test, turn_dense=turn_dense, cat_split=59)
    
    # baseline accuracy score
    null_score = np.max([y_test.mean(), 1 - y_test.mean()])
    print('accuracy baseline:', null_score)

    # ---------- calculate y_pred using log. reg. and stochastic gradient descent
    if do_logreg:
        
        M = 100
        n_epochs = 400

        logistic_regression(X_train, X_test, y_train, y_test, 
                                M = M, n_epochs=n_epochs, plot_cost=True, save=True)


    # ------------ neural network
    if do_NN:
        print('\n ------- Neural network regession --------')
        epochs = 200
        batch_size = 100
        eta = 0.001
        lmbd = 1e-5
        n_categories = 2
        n_hidden_layers = 1


        n_hidden_neurons1 = int(np.floor(np.mean(n_categories + X_train.shape[1])))
        print('No of hidden neurons:',n_hidden_neurons1)
        
        n_hidden_neurons2 = int(np.round(X_train.shape[0]/(8*(n_categories+X_train.shape[1]))))
        print('No of hidden neurons:',n_hidden_neurons2)
        
        n_hidden_neurons3 = int(np.round(2./3*X_train.shape[1]) + n_categories)
        print('No of hidden neurons:',n_hidden_neurons3)

        if do_NN_np:
            epochs = 1000
            batch_size = 100
            
            
            n_hidden_neurons = n_hidden_neurons3
            hidden_layer_sizes = n_hidden_layers*[n_hidden_neurons,]
            #hidden_layer_sizes = [n_hidden_neurons, int(np.round(n_hidden_neurons/2))]
            print('No of hidden neurons:', hidden_layer_sizes)
            init_method = 'Xavier'
            out_activation = 'softmax'
            hidden_activation = 'sigmoid'
            cost_f = 'ce'

            y_train_onehot = OneHotEncoder(categories='auto').fit_transform(y_train).toarray()
            

            #eta_vals = np.array([eta])
            #lmbd_vals = np.array([lmbd])

            eta_vals = np.logspace(-6, -1, 6)
            lmbd_vals = np.logspace(-6, -1, 6)
            
            train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
            train_auc = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_auc = np.zeros((len(eta_vals), len(lmbd_vals)))

            # store the models for later use
            DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
            
            for i, eta in enumerate(eta_vals):
                for j, lmbd in enumerate(lmbd_vals):
                    print('eta:',eta, 'lambda:', lmbd)
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

            arg_max = np.unravel_index(test_accuracy.argmax(), test_accuracy.shape)
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
                
                print('Accuracy baseline:', null_score)
                print("Accuracy score NN: ", accuracy_score(y_test, y_pred_NN))
                print('f1 score NN:', f1_score(y_test, y_pred_NN))
                print('AUC CGC:', auc_NN_test)
                print('Area ratio:', area_ratio_NN_test)

                if plot_cumgain_test:
                    plot_cum_gain_chart(y_test, prob_y_NN)
                if plot_cumgain_train:
                    plot_cum_gain_chart(y_train, prob_y_NN_train)
                
            elif vizualize_scores:

                tools.vizualize_scores(test_auc, title='', xticks=lmbd_vals, 
                                    yticks=eta_vals, xlab=r'$\lambda$', ylab=r'$\eta$')

        if do_NN_sk:
            # --------- NN with scikit learn
            print('\n ---------- Neural Network classification sklearn')
            
            batch_size = 100
            epochs=1000
            
            eta_vals = np.array([eta])
            lmbd_vals = np.array([lmbd])


            #eta_vals = np.logspace(-7, 0, 8)
            #lmbd_vals = np.logspace(-7, 0, 8)

            # store models for later use
            DNN_sk = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
            
            
            train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
            train_auc = np.zeros((len(eta_vals), len(lmbd_vals)))
            test_auc = np.zeros((len(eta_vals), len(lmbd_vals)))
            
            n_hidden_neurons = n_hidden_neurons3
            hidden_layer_sizes = n_hidden_layers*(n_hidden_neurons,) # (neurons,neurons,...)
            print('No of hidden neurons:', hidden_layer_sizes)

            for i, eta in enumerate(eta_vals):
                for j, lmbd in enumerate(lmbd_vals):
                    print('eta:',eta, 'lambda:', lmbd)
    
                    dnn_sk = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='logistic',
                                        alpha=lmbd, solver='adam', learning_rate_init=eta, max_iter=epochs)
                    dnn_sk.fit(X_train, y_train.ravel())
                    #y_pred_NN_sk = dnn_sk.predict(X_test)
                    #prob_y_NN_sk = dnn_sk.predict_proba(X_test)
            
            
                    DNN_sk[i][j] = dnn_sk
        
                    y_pred_NN_sk = dnn_sk.predict(X_test)
                    y_pred_NN_sk_train = dnn_sk.predict(X_train)
                    
                    train_accuracy[i][j] = accuracy_score(y_train, y_pred_NN_sk_train)
                    test_accuracy[i][j] = accuracy_score(y_test,y_pred_NN_sk)
                    #auc_train, area_ratio_train = auc_CGC(y_train, prob_y_NN_sk_train)                    
                    train_auc[i][j] = f1_score(y_train,y_pred_NN_sk_train)
                    test_auc[i][j] = f1_score(y_test,y_pred_NN_sk)

            arg_max = np.unravel_index(test_auc.argmax(), test_auc.shape)
            print('arg_max:', arg_max)
            print('best param: eta:', eta_vals[arg_max[0]], 'lambda:',lmbd_vals[arg_max[1]])
            print('best accuracy test score:', test_accuracy[arg_max])
            print('best accuracy train score:', train_accuracy[arg_max])
            print('best f1 test score:', test_auc[arg_max])
            print('best f1 train score:', train_auc[arg_max])

            if (len(eta_vals)< 2 and len(lmbd_vals<2)):

                prob_y_NN_sk = dnn_sk.predict_proba(X_test)
                prob_y_NN_sk_train = dnn_sk.predict_proba(X_train)
                auc_NN_sk_test, area_ratio_NN_sk_test = auc_CGC(y_test, prob_y_NN_sk)


                print('ytest sum:', y_test.sum())
                print('hidden layers:', n_hidden_layers)
                print('yhat NN sum:', y_pred_NN_sk.sum())
                print('confusion test :', confusion_matrix(y_test, y_pred_NN_sk))
                print('confusion train:', confusion_matrix(y_train, y_pred_NN_sk_train))
                
                print("Accuracy score NN: ", accuracy_score(y_test, y_pred_NN_sk))
                print('f1 score NN:', f1_score(y_test, y_pred_NN_sk))
                print('AUC CGC:', auc_NN_sk_test)
                print('Area rato:', area_ratio_NN_sk_test)

                if plot_cumgain_test:
                    plot_cum_gain_chart(y_test, prob_y_NN, prob_y_NN_sk)
                if plot_cumgain_train:
                    plot_cum_gain_chart(y_train, prob_y_NN_sk_train)

            elif vizualize_scores:
                tools.vizualize_scores(test_auc, title='', xticks=lmbd_vals, 
                                    yticks=eta_vals, xlab=r'$\lambda$', ylab=r'$\eta$')


main_logreg(remove_pay0=True, resample=True, turn_dense=False, do_logreg=False, 
            do_NN=True, do_NN_np=True, do_NN_sk=False, 
            plot_cumgain_test=False, vizualize_scores=True)
