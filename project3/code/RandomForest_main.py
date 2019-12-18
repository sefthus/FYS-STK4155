#import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, f1_score, precision_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE

import tools


def create_data(filename=r'./data/pulsar_stars.csv'):
    """
        Creates a dataframe from filename using pandas. Removes invalid values
        from the factors. 
        Splits the dataframe into the design matrix X consisting of the features,
        and the respone y.
        Returns the design matrix X and respone y.

        filename  : the filename of the dataframe
        resample: if True, SMOTE resampling is used to balance the data
    """
    nanDict = {}

    df = pd.read_csv(filename, header=0, skiprows=0, na_values=nanDict)
    #df.rename(index=str, columns={"target_class":"class"}, inplace=True)
    df.columns = ['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile', 'mean_dmsnr',
               'std_dmsnr', 'kurtosis_dmsnr', 'skewness_dmsnr', 'class']
    
    print('no of features: %s' %df.shape[1])
    print('no of examples: %s' %df.shape[0])
    
    #print(df.head())

    print('no of missing values')
    print(df.isnull().sum())
    
    # features and targets
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    return X, y

def plot_metrics(x_train, y_train, x_test, y_test, train_label='train', test_label='test', xlabel=' ', ylabel=' ', lw=2, labelsize=14, ticksize=12, legendsize=12):
    plt.plot(x_train, y_train, label=train_label, linewidth=lw)
    
    if x_test is not None:
        plt.plot(x_test, y_test, label=test_label, linewidth=lw)
        plt.legend(prop={'size': legendsize})
    
    plt.xlabel(xlabel, size=labelsize)
    plt.ylabel(ylabel, size=labelsize)
    plt.grid('True', linestyle='dashed')
    plt.tick_params(axis='both', labelsize=ticksize)
    plt.tight_layout()
    plt.show()

def fit_rf(X_train, X_test, y_train, y_test, resample=0, use_weights=0, bootstrap=False, max_depth=None, max_features='auto', min_samples_leaf=1, min_samples_split= 2, n_estimators=500):
    
    weights=None
    if use_weights:
            weight='balanced'

    rf = RandomForestClassifier(
                                n_estimators=n_estimators, bootstrap=bootstrap, max_depth=max_depth,
                                max_features=max_features, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                n_jobs=1, random_state=seed+1, oob_score=True, class_weight=weights
                                )
    model = rf
    if resample:
        model = make_pipeline(SMOTE(0.25, random_state=seed), 
                                rf)
        print('CV score train:', np.mean(cross_val_score(model, X_train, y_train, scoring='f1', cv=5, n_jobs=20, verbose=0)))


    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    print('training metrics')
    print('accuracy train:', accuracy_score(y_train, y_train_pred))

    if not resample:

        y_train_oob = np.argmax(model.oob_decision_function_, axis=1)
        print('oob training metrics')
        print('accuracy oob:', model.oob_score)
        print('accuracy oob train:', accuracy_score(y_train, y_train_oob))
        print(classification_report(y_train, y_train_oob))
    y_test_pred = model.predict(X_test)
    
    #print('training metrics')
    #print('accuracy train:', accuracy_score(y_train, y_train_pred))
    #print(classification_report(y_train, y_train_pred))
    

    print('testing metrics')
    print('accuracy test:', accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

def rf_default(X_train, X_test, y_train, y_test):
    def_model = RandomForestClassifier(n_estimators = 100, random_state = seed)
    def_model.fit(X_train, y_train)
    def_f1 = f1_score(y_test, def_model.predict(X_test))

    def_pred = def_model.predict(X_train)
    print('accuracy', accuracy_score(y_train, def_pred))
    print('f1: %f using default params' % def_f1)
    return def_f1

def rf_tuning(X_train, X_test, y_train, y_test, resample=0, use_weights=0):


    weights=None
    if use_weights:
            weights='balanced'
    rf = RandomForestClassifier(class_weight=weights)#, oob_score=True)
    model = rf
    
    n_estimators = [100, 300, 500, 700, 1000]
    max_features = [2, 3, 4, 'auto']
    max_depth = [5, 7, 10, 20, 30, None]
    min_samples_split = [5, 10, 20]
    min_samples_leaf = [2, 3, 5, 10]
    bootstrap = [True]

    param_grid = {
                    'n_estimators': n_estimators
                    , 'max_features': max_features
                    , 'max_depth': max_depth
                    , 'min_samples_split': min_samples_split
                    , 'min_samples_leaf': min_samples_leaf
                    , 'bootstrap': bootstrap
                }
    
    if resample:
        model = Pipeline([
            ('sampling', SMOTE(1, random_state=seed+2)),
            ('clf', rf)
        ])
        param_grid = {'clf__' + key: param_grid[key] for key in param_grid}

    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=30, cv=3, scoring='f1', verbose=2, refit=True)
    grid_result = grid.fit(X_train, y_train)

    def_f1 = rf_default(X_train, X_test, y_train, y_test)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    print('Improvement of %.2f' %( 100*(grid_result.best_score_ - def_f1)/def_f1))
    
    print('f1 validation:', grid_result.best_score_)
    print('f1 test:      ', f1_score(y_test, grid.predict(X_test)))
    #y_pred_test = grid.predict(X_test)
    #y_pred_train = grid.predict(X_train)

    #print(classification_report(y_pred_test, y_test))
    #print(classification_report(y_pred_train, y_train))

def rf_random_tuning(X_train, X_test, y_train, y_test, resample=0, use_weights=0):

    weights=None
    if use_weights:
            weights='balanced'

    rf = RandomForestClassifier(class_weight=weights)
    model = rf

    n_estimators = np.arange(100, 2200, 200).tolist()
    max_features = [2, 3, 4, 'auto']
    max_depth = [3, 5, 10, 20, 40, 90, None]
    min_samples_split = [2, 5, 10, 20]
    min_samples_leaf = [2, 4, 6, 8, 15]
    bootstrap = [True]

    param_grid = {
                    'n_estimators': n_estimators
                    , 'max_features': max_features
                    , 'max_depth': max_depth
                    , 'min_samples_split': min_samples_split
                    , 'min_samples_leaf': min_samples_leaf
                    , 'bootstrap': bootstrap
                }
    if resample:
        model = Pipeline([
            ('sampling', SMOTE(0.25, random_state=seed+2)),
            ('clf', rf)
        ])
        param_grid = {'clf__' + key: param_grid[key] for key in param_grid}


    rgrid_result = RandomizedSearchCV(estimator = model, param_distributions = param_grid, n_iter = 300, scoring='f1',cv = 3, verbose=2, random_state=seed, n_jobs = 25)
    rgrid_result.fit(X_train, y_train)

    def_f1 = rf_default(X_train, X_test, y_train, y_test)
    print("Best: %f using %s" % (rgrid_result.best_score_, rgrid_result.best_params_))
    print('Improvement of %.2f' %( 100*(rgrid_result.best_score_ - def_f1)/def_f1))


if __name__ == '__main__':

    seed = 7
    np.random.seed(seed)

    X, y = create_data()
    print('Ratio true pulsars to false pulsars:', np.sum(y)/(len(y)-np.sum(y)))
    print('Ratio true pulsars to total datapoints:', np.sum(y)/len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)
    #X_train, X_test = tools.scale_data(X_train, X_test, cat_split=None)

    print('Ratio pulsars to false pulsars:')
    print('   test set:      ', np.sum(y_test)/(len(y_test)-np.sum(y_test)))
    print('   train set:     ', np.sum(y_train)/(len(y_train)-np.sum(y_train)))


    #rf_tuning(X_train, X_test, y_train, y_test, resample=0, use_weights=0)
    #rf_random_tuning(X_train, X_test, y_train, y_test, resample=0, use_weights=0)

    fit_rf(X_train, X_test, y_train, y_test, resample=1, bootstrap=True, max_depth=15, max_features=2, min_samples_leaf=4, min_samples_split=10, n_estimators=500)

