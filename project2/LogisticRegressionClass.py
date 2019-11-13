import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, SGDRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from scipy.special import xlogy, xlog1py, expit
import sys

np.random.seed(3155)
class LogisticRegressor:

    def __init__(self, X_train, y_train):
            self.X_train = X_train
            self.y_train = y_train
            #self.model_key = model


    def fit_sklearn(self, lambdas = None, solver="lbfgs", refit ='roc_auc'):
        # --------- Logistic regression
        logReg = LogisticRegression(max_iter=1e4, class_weight='balanced')

        if lambdas == None:
            lambdas = np.logspace(-5, 7, 13)
        
        parameters = [{'C':1/lambdas, 'solver':[solver]}]
        
        scoring =['accuracy', 'roc_auc']
        self.model = GridSearchCV(logReg, parameters, cv=5, scoring=scoring, refit=refit)
        self.model.fit(self.X_train, self.y_train.ravel())

        self.model
    
    def predict_sklearn(self, X):
        #model = self.logreg_sklearn()
        y_pred = self.model.predict(X)
        return y_pred

    def predict_probabilities_sklearn(self, X):
        #model = self.logreg_sklearn()
        probabilities = self.model.predict_proba(X)
        return probabilities

    def fit_numpy(self, M=40, n_epochs=200):
        """
            Performs a logistic regression of a binary respinse, using 
            stochastic gradient descent. Calculates the regression parameters beta.

            M:         points in each batch, to be used in the SGD
            n_epochs:  number of epochs used in SGD
            plot_cost: wether or not to plot the cost function, passed onto SGD solver

        """
        self.n_epochs = n_epochs

        self.beta = self.stochastic_gd(M, self.n_epochs)# , M=len(self.X_train)) # =gradient descent
        
        
    def predict_probabilities_numpy(self, X):
        """
            returns probability of y_pred being in class 0 (col 0) and class 1 (col 1)
        """

        prob_y = 1/(1 + np.exp(-X.dot(self.beta)))

        return np.column_stack([1-prob_y, prob_y])

    def predict_numpy(self, X):
        """
            returns y_predicted
        """
        prob_y = self.predict_probabilities_numpy(X)[:,1]

        # round up or down y_pred to get on binary form
        y_pred = np.zeros(len(prob_y))
        y_pred[prob_y >= 0.5] = 1
        y_pred[prob_y < 0.5] = 0

        return y_pred

    def stochastic_gd(self, M, n_epochs):
        """ 
            Calculates the regression parameters beta using stochastic
            gradient descent. 
            Returns beta

            M:        number of minibatches
            n_epochs: number of epochs
            plot_cost: if True, plot the epochs against the cost function
        """

        n =  self.X_train.shape[0] # datapoints
        m = int(n/M) # number of minibatches
        t0 = 1.
        t1 = 10.

        self.cost_epoch = np.zeros(self.n_epochs)
        beta = np.random.randn(self.X_train.shape[1], 1) # initial beta parameters

        eta_j = t0/t1 # initial learning rate # 
        #eta_j= 0.001 # use as constant

        for epoch in range(1, self.n_epochs+1):
            X, y = shuffle(self.X_train, self.y_train, random_state=0)
            
            for k in range(m):
                #k = i#np.random.randint(m) # pick random kth minibatch
                Xk = X[k*M:(k+1)*M,:]
                yk = y[k*M:(k+1)*M]

                if k == m-1: 
                    Xk = X[k*M:,:]
                    yk = y[k*M:,:]

                # compute gradient  and cost log reg
                sigmoid = expit(Xk.dot(beta))
                sigmoid_min = expit(-Xk.dot(beta)) # =1-sigmoid(x) = sigmoid(-x) 
                
                self.cost_epoch[epoch-1] += -np.sum( xlogy(yk, sigmoid) + xlogy((1-yk),sigmoid_min))
                
                #self.cost_epoch[epoch-1] += -np.sum( yk*np.log(sigmoid) + (1-yk)*np.log(sigmoid_min) )
                
                gradient = - Xk.T.dot(yk - sigmoid)

                # compute new beta
                t = epoch*m + k
                eta_j = t0/(t+t1) # adaptive learning rate

                beta = beta - eta_j*gradient
                #sys.exit()
        return beta

    def plot_cost_function(self, save=False, name='cost.png'):
    
        plt.plot(np.linspace(1, self.n_epochs+1, self.n_epochs), self.cost_epoch, linewidth=2)
        plt.xlabel('Epoch', size=14)
        plt.ylabel('Cost function',size=14)
        plt.grid('True', linestyle='dashed')
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()

        if save:
            plt.savefig(r'.\figures'+ name +'.png')
            plt.savefig(r'.\figures'+ name +'.pdf')

        plt.show()
