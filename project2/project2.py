
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
from scipy.stats import percentileofscore
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, mean_squared_error, r2_score, f1_score
from scikitplot.metrics import plot_cumulative_gain
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier

from NNclass import NeuralNetwork
from original_NNclass import NeuralNetwork as oNeuralNetwork
from NNclass_arr import NeuralNetwork as aNeuralNetwork
seed = 3155
np.random.seed(seed)

# reading file into data frame

def drop_value(dframe, variables, val):
    """ 
        drop rows which have values equal to val

        dframe: data frame
        vars  : data frame variables, i.e., dframe.PAY_0
        val   : value to be dropped
        
    """
    #print('hello')
    #if any(isinstance(i, list) for i in variables):
    #if variables.ndim==1:
        #print(dframe[(variables == val)].index)
    dframe = dframe.drop(dframe[(variables == val)].index)
    #else:
    #    old_remove_idx = []
    #    dframe_old = dframe
    #    for var in variables:
    #        remove_idx=dframe_old[(var == val)].index
    #        print(len(remove_idx))
    #        remove_idx = list(filter(lambda i: i not in old_remove_idx, remove_idx))
    #        print(len(remove_idx))
    #        dframe = dframe.drop(remove_idx, axis=0)
    #        old_remove_idx=remove_idx
    
    return dframe

def pay_remove_value(dframe, value):
    dframe = drop_value(dframe, dframe.PAY_0, value)
    dframe = drop_value(dframe, dframe.PAY_2, value)
    dframe = drop_value(dframe, dframe.PAY_3, value)
    dframe = drop_value(dframe, dframe.PAY_4, value)
    dframe = drop_value(dframe, dframe.PAY_5, value)
    dframe = drop_value(dframe, dframe.PAY_6, value)
    return dframe

def education_remove_value(dframe, value):
    
    dframe = drop_value(dframe, dframe.EDUCATION, 5)
    dframe = drop_value(dframe, dframe.EDUCATION, 6)
    dframe = drop_value(dframe, dframe.MARRIAGE, 0)
    return dframe

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

        df = education_remove_value(df,value=0)
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

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

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

def scale_data(X_train, X_test, cat_split=60, turn_dense=False):
    """
        Scales the training and test data with StandardScaler() from scikit learn.
        The categorical data should already be onehotencoded.
        The data is first split into numerical and categorical data according to
        cat_split, and then the numerical data is scaled.
        Returns the scaled training and test data.

        X_train:    the training data
        X_test:     the test data
        cat_split:  where to split the data into categorical and numerical arrays
        turn_dense: if the data is a sparse matrix, setting this to True, 
                    will transform the data to dense numpy arrays 
    """

    if sparse.issparse(X_train):
        if turn_dense:
            X_train, X_test = X_train.toarray(), X_test.toarray()

        else:
            X_train_cat, X_train_num = X_train[:, :cat_split], X_train[:, cat_split:].toarray()
            X_test_cat, X_test_num = X_test[:, :cat_split], X_test[:, cat_split:].toarray()

            scaleX = StandardScaler().fit(X_train_num)
            
            X_train_num_scale = sparse.csr_matrix(scaleX.transform(X_train_num))
            X_test_num_scale = sparse.csr_matrix(scaleX.transform(X_test_num))
        
            X_train = sparse.hstack([X_train_cat, X_train_num_scale]).tocsr()
            X_test = sparse.hstack([X_test_cat, X_test_num_scale]).tocsr()
    
    if not sparse.issparse(X_train):
        
        X_train_cat, X_train_num = X_train[:, :cat_split], X_train[:, cat_split:]
        X_test_cat, X_test_num = X_test[:, :cat_split], X_test[:, cat_split:]

        scaleX = StandardScaler().fit(X_train_num)
        
        X_train_num_scale = scaleX.transform(X_train_num)
        X_test_num_scale = scaleX.transform(X_test_num)

        X_train = np.concatenate((X_train_cat, X_train_num_scale), axis=1)
        X_test = np.concatenate((X_test_cat, X_test_num_scale), axis=1)        

    return X_train, X_test

def accuracy_score_func(y_pred, y_test):
    return np.sum(y_test == y_pred) / len(y_test)

def logreg_sklearn(X, y, no_grid=False):
    # --------- Logistic regression
    logReg = LogisticRegression(max_iter=1e4)
    y = y.ravel()
    if no_grid:
        logReg.fit(X, y)

        return logReg

    else:
        lambdas = np.logspace(-5, 7, 13)
        parameters = [{'C':1/lambdas, "solver":["lbfgs"]}]
        scoring =['accuracy', 'roc_auc']
        gridSearch = GridSearchCV(logReg, parameters,cv=5, scoring=scoring, refit='roc_auc')
        gridSearch.fit(X, y)

        return gridSearch

def bestCurve(y):
    defaults = np.sum(y == 1)
    total = len(y)
    x = np.linspace(0, 1, total)
    y1 = np.linspace(0, 1, defaults)
    y2 = np.ones(total-defaults)
    y3 = np.concatenate([y1,y2])
    return x, y3

def plot_cum_gain_chart(y_true, prob_y, prob_ysk=None):

    if prob_ysk is not None:
        ax = plot_cumulative_gain(y_true, prob_y)
        plot_cumulative_gain(y_true, prob_ysk, ax=ax) # they overlap!
    else:
        plot_cumulative_gain(y_true, prob_y)
    x, y = bestCurve(y_true)
    plt.plot(x, y, label='best curve')
    plt.legend()
    plt.show()


# DONE: regression: linear, classification:softmax, not sigmoid in last layer???
# 1 node=neuron i siste lag når regresjon
def main(remove_nan=True, turn_dense=False, do_logreg=True, do_NN=True, plot_cumgain_test=False, plot_cumgain_train=False):
    
    X, y = create_df(remove_nan=remove_nan)

    training_share = 0.7
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-training_share, shuffle=True, random_state=seed)

    X_train, X_test = scale_data(X_train, X_test, turn_dense=turn_dense)
    
    # ---------- calculate y_pred using log. reg. and stochastic gradient descent
    if do_logreg:
        beta = stochastic_gd(X_train, y_train, M=40, n_epochs=200)# , M=len(X_train)) # =gradient descent

        # probabilities of non-default (0) and default (1)    
        y_pred = 1/(1 + np.exp(-X_test.dot(beta)))
        prob_y = np.column_stack([1-y_pred, y_pred])

        y_pred_train = 1/(1 + np.exp(-X_train.dot(beta)))
        prob_y_train = np.column_stack([1-y_pred_train, y_pred_train])

        # round up or down y_pred to get on binary form
        y_pred_b = np.zeros_like(y_pred)
        y_pred_b[y_pred >= 0.5] = 1
        y_pred_b[y_pred < 0.5] = 0
        
        print('ytest sum:', np.sum(y_test))
        print('yhat sum:', np.sum(y_pred_b)) # ensure not only predicting zero's
        print('yhat shape:', y_pred_b.shape)
        print('accuracy score:', accuracy_score(y_test, y_pred_b))
        print('f1 score:', f1_score(y_test, y_pred_b))

        # trying to make cumulative gain chart. sort and cumsum
        arry_pred_sort = np.squeeze(y_pred.argsort(axis=0))[::-1]
        def_cum = np.cumsum(y_pred_b[arry_pred_sort])/np.sum(y_pred_b)
        
        
        # -------- calcuate y_pred using sklearn's logistic regression
        model = logreg_sklearn(X_train, y_train)
        y_pred_sk = model.predict(X_test)
        print('yhat sklearn sum:', np.sum(y_pred_sk))
        print('accuracy score sklearn:', accuracy_score(y_test, y_pred_sk))
        print('f1 score sklearn:', f1_score(y_test, y_pred_sk))

        # probabilities of non-default 0 and default 1
        prob_ysk = model.predict_proba(X_test)
        prob_ysk_train = model.predict_proba(X_train)
        '''
        # trying to make cumulative fain chart
        arr_ysk = np.squeeze(prob_ysk[:,1].argsort(axis=0))[::-1] # descending
        def_sort = y_pred_sk[arr_ysk]
        prob_sort = prob_ysk[arr_ysk]
        def_split = np.array_split(def_sort.ravel(), 100)

        summed_arr = [np.sum(arr) for arr in def_split]
        cumsum_arr = np.cumsum(summed_arr)/np.sum(summed_arr)
        cumsum_arr = np.append( 0, cumsum_arr)
        '''
        if plot_cumgain_test:
            plot_cum_gain_chart(y_test, prob_y)#, prob_ysk)
            print('prob shape:', prob_y.shape)
        
        if plot_cumgain_train:
            plot_cum_gain_chart(y_train, prob_y_train, prob_ysk_train)
        #plt.plot(np.linspace(0,1,len(cumsum_arr)),cumsum_arr, label='test3')
        #x1 = np.linspace(0,1, len(def_cum))
        #plt.plot(x1, pred_def, label='test')


    # ------------ neural network
    if do_NN:
        print('\n ------- Neural network regession --------')
        epochs = 30
        batch_size = 500
        eta = 1e-4
        lmbd = 0.
        n_categories = 2 #=1 when lin. reg
        n_hidden_layers = 1
        n_hidden_neurons = 50
        hidden_layer_sizes = n_hidden_layers*[n_hidden_neurons,]
        init_method='random'
        out_activation = 'softmax'
        cost_f = 'sigmoid'

        y_train_onehot = OneHotEncoder(categories='auto').fit_transform(y_train).toarray()
        #y_train_onehot = to_categorical_numpy(y_train)

        #eta_vals = np.array([1e-4])
        eta_vals = np.logspace(-7, -1, 7)
        #lmbd_vals = np.array([0])
        lmbd_vals = np.logspace(-7, 0, 8)
        
        train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
        test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
        train_auc = np.zeros((len(eta_vals), len(lmbd_vals)))
        test_auc = np.zeros((len(eta_vals), len(lmbd_vals)))

        # store the models for later use
        DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
        
        for i, eta in enumerate(eta_vals):
            for j, lmbd in enumerate(lmbd_vals):
                print('eta:',eta, 'lambda:', lmbd)
                #dnn = aNeuralNetwork(X_train, y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                #                    n_hidden_neurons=hidden_layer_sizes, n_categories=n_categories, init_method=init_method)

                dnn = oNeuralNetwork(X_train, y_train_onehot, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
                                    n_hidden_neurons=n_hidden_neurons, n_categories=n_categories, init_method=init_method,
                                    out_activation = out_activation, cost_f = cost_f )

                dnn.train()

                DNN_numpy[i][j] = dnn
    
                y_pred_NN = dnn.predict(X_test)
                y_pred_NN_train = dnn.predict(X_train)

                train_accuracy[i][j] = accuracy_score(y_train, y_pred_NN_train)
                test_accuracy[i][j] = accuracy_score(y_test,y_pred_NN)
                train_auc[i][j] = roc_auc_score(y_train,y_pred_NN_train)
                test_auc[i][j] = roc_auc_score(y_test,y_pred_NN)

                #sys.exit()
        '''
        prob_y_NN = dnn. predict_probabilities(X_test)
        prob_y_NN_train = dnn.predict_probabilities(X_train)

        print('ytest sum:', y_test.sum())
        print('hidden layers:', n_hidden_layers)
        print('yhat NN sum:', y_pred_NN.sum())
        #sys.exit()
        # accuracy score from scikit library
        print("Accuracy score NN: ", accuracy_score(y_test, y_pred_NN))
        print('f1 score NN:', f1_score(y_test, y_pred_NN))'''
                
        '''
        fig, ax = plt.subplots(figsize = (10, 10))
        #sns.heatmap(train_accuracy, xticklabels=lmbd_vals, yticklabels=eta_vals, annot=True, ax=ax, cmap="viridis")
        #ax.set_title("Training Accuracy")

        sns.heatmap(train_auc, xticklabels=lmbd_vals, yticklabels=eta_vals, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Training ROC AUC")
        
        ax.set_ylabel(r"$\eta$")
        ax.set_xlabel(r"$\lambda$")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()
        '''
        fig, ax = plt.subplots(figsize = (8, 8))
        #sns.heatmap(test_accuracy, xticklabels=lmbd_vals, yticklabels=eta_vals, annot=True, ax=ax, cmap="viridis")
        #ax.set_title("Test Accuracy")
        sns.heatmap(test_auc, xticklabels=lmbd_vals, yticklabels=eta_vals, annot=True, ax=ax, cmap="viridis")
        ax.set_title("Test ROC AUC")
        ax.set_ylabel(r"$\eta$")
        ax.set_xlabel(r"$\lambda$")
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.show()

        sys.exit()
        # --------- NN with scikit learn
        # store models for later use
        #DNN_scikit = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

        hidden_layer_sizes = n_hidden_layers*(n_hidden_neurons,) # (neurons,neurons,...)
        dnn_sk = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation='logistic',
                            alpha=lmbd, learning_rate_init=eta, max_iter=epochs)
        dnn_sk.fit(X_train, y_train.ravel())
        y_pred_NN_sk = dnn_sk.predict(X_test)
        prob_y_NN_sk = dnn_sk.predict_proba(X_test)
        
        
        #print("Accuracy score on test set: ", dnn_sk.score(X_test, y_test.ravel()))
        print('yhat NN sum sklearn:', y_pred_NN_sk.sum())
        print("Accuracy score NN sklearn: ", accuracy_score(y_test, y_pred_NN_sk))
        print('f1 score NN sklearn:', f1_score(y_test, y_pred_NN_sk))
        #sys.exit()
        # Things are looking good! same results sk and mine

        if plot_cumgain_test:
            #print('prob y:', prob_y_NN.shape)
            plot_cum_gain_chart(y_test, prob_y_NN)


main(remove_nan=True, turn_dense=False, do_logreg=False, do_NN=True, plot_cumgain_test=False)