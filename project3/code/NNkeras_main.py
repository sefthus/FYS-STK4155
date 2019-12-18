#import collections
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, f1_score, precision_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from imblearn.pipeline import Pipeline, make_pipeline
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers import SGD

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

def SMOTE_resample(X, y, min_to_maj_ratio=0.25):

    sm = SMOTE(sampling_strategy=min_to_maj_ratio, random_state=seed+2)
    X, y = sm.fit_resample(X, y.ravel())

    return X, y

def create_NN(
                optimizer='adam'
                #, learning_rate=0.01
                , activation='relu'
                , neurons1=10
                , neurons2=5
                , lmbd=0
                , dropout_rate=0
                , init_weight = 'normal'
                ):
    
    model = Sequential()
    model.add(layers.Dense(neurons1, input_dim=X.shape[1], activation=activation
    #                    , kernel_regularizer=regularizers.l2(lmbd)
                        , kernel_initializer=init_weight
                        , bias_initializer=init_weight
                        ))
    model.add(layers.Dropout(dropout_rate))
    if neurons2 is not None:
        model.add(layers.Dense(neurons2, activation=activation
        #                    , kernel_regularizer=regularizers.l2(lmbd)
                            , kernel_initializer=init_weight
                            , bias_initializer=init_weight
                            ))
        model.add(layers.Dropout(dropout_rate))
    
    model.add(layers.Dense(1, activation = 'sigmoid')) # output layer binary classification


    #optimizer = SGD(learning_rate=learning_rate, momentum=momentum)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def plot_metrics(x_train, y_train, x_test, y_test, train_label='train', test_label='test', xlabel=' ', ylabel=' ', lw=2, labelsize=14, ticksize=12, legendsize=12):
    plt.plot(x_train, y_train, label=train_label, linewidth=lw)
    
    if x_test is not None:
        plt.plot(x_test, y_test, '--', label=test_label, linewidth=lw)
        plt.legend(prop={'size': legendsize})
    
    plt.xlabel(xlabel, size=labelsize)
    plt.ylabel(ylabel, size=labelsize)
    plt.grid('True', linestyle='dashed')
    plt.tick_params(axis='both', labelsize=ticksize)
    plt.tight_layout()
    plt.show()

def fit_NN(X_train, X_val, X_test, y_train, y_val, y_test, resample=0, use_weights=0, epochs=500, batch_size=50, optimizer='sgd', activation='relu', n1=10, n2=5, lmbd=0.01, dropout_rate=0):
    
    model = create_NN(optimizer=optimizer, activation=activation, neurons1=n1, neurons2=n2, lmbd=lmbd, dropout_rate=dropout_rate)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
    
    history = None
    weights=None

    if use_weights:
            weights=compute_class_weight('balanced', np.unique(y_train), y_train)

    if resample:
        X_res, y_res = SMOTE_resample(X_train, y_train, 0.25)
        history = model.fit(X_res, y_res, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val) , verbose=2, callbacks=[es])
    
    else:
        history = model.fit(X_train, y_train, class_weight=weights, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val) , verbose=2, callbacks=[es])


    epochs_used = np.linspace(1,len(history.history['accuracy']), len(history.history['accuracy']))

    plot_metrics(epochs_used, history.history['accuracy'], epochs_used, history.history['val_accuracy']
                , train_label='train', test_label='validation', xlabel='epochs', ylabel='accuracy')
    
    plot_metrics(epochs_used, history.history['loss'], epochs_used, history.history['val_loss']
                , train_label='train', test_label='validation', xlabel='epochs', ylabel='cross-entropy loss')


    null_score = np.max([y_test.mean(), 1 - y_test.mean()])
    print('accuracy baseline:', null_score)

    y_train_pred = model.predict_classes(X_train, verbose=0)

    y_test_pred = model.predict_classes(X_test, verbose=0)

    print('training metrics')
    print('accuracy train:', accuracy_score(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred))

    print('testing metrics')
    print('accuracy test:', accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred))

def nn_tuning(X_train, y_train, resample=0, use_weights=0):

    kerasNN = KerasClassifier(build_fn=create_NN
                            , epochs=300
                            , batch_size=64
                            #, neurons1=15
                            #, neurons2=5
                            , optimizer='Nadam'
                            , lmbd = 1e-4
                            , dropout_rate = 0.1
                            , verbose=2
                            )

    model = kerasNN

    batch_size = [16, 64, 128]
    epochs = [50, 100, 200, 300, 400, 500]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    neurons1 = [5, 10, 15, 20, 30]
    neurons2 = [2, 5, 10, 20, 30]
    lmbd = [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    init_weight = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']    
    weight_constraint = [1, 2, 3, 4, 5]
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    param_grid = {
                        #'batch_size': batch_size,
                        #'epochs': epochs
                        #'optimizer': optimizer,
                        #'learning_rate': learning_rate,
                        #'momentum': momentum,
                        'neurons1': neurons1,
                        'neurons2': neurons2,
                        #'init_weight': init_weight, # normal is best, one layer
                        #'dropout_rate' dropout_rate,
                        #'weight_constraint': weight_constraint
                        #'lmbd': lmbd
                }

    if resample:
        model = Pipeline([
            ('sampling', SMOTE(0.25, random_state=seed+2)),
            ('clf', kerasNN)
        ])
        param_grid = {'clf__' + key: param_grid[key] for key in param_grid}

    weights=None
    if use_weights:
            weights=compute_class_weight('balanced', np.unique(y_train), y_train)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=20, 
                        cv=3, scoring='f1', verbose=2)
    #sys.exit()
    grid_result = grid.fit(X_train, y_train, class_weight=weights)
    # if slow, fit on subset of data


    #summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

def nn_random_tuning(X_train, y_train, resample=0, use_weights=0):

    kerasNN = KerasClassifier(build_fn=create_NN
                            #, epochs=300
                            #, batch_size=64
                            , optimizer='Nadam'
                            , verbose=2
                            )

    model = kerasNN

    batch_size = [16, 64, 128]
    epochs = [50, 100, 200, 300, 400, 500]
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #learning_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    #momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
    neurons1 = [5, 10, 15, 20, 30]
    neurons2 = [2, 5, 10, 20, 30]
    lmbd = [0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
    init_weight = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']    
    dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    param_grid = {
                        'batch_size': batch_size,
                        'epochs': epochs,
                        #'optimizer': optimizer,
                        #'learning_rate': learning_rate,
                        #'momentum': momentum,
                        'neurons1': neurons1,
                        'neurons2': neurons2,
                        #'init_weight': init_weight, # normal is best, one layer
                        'dropout_rate': dropout_rate,
                        #'lmbd': lmbd
                }

    if resample:
        model = Pipeline([
            ('sampling', SMOTE(0.25, random_state=seed+2)),
            ('clf', kerasNN)
        ])
        param_grid = {'clf__' + key: param_grid[key] for key in param_grid}

    weights=None
    if use_weights:
            weights=compute_class_weight('balanced', np.unique(y_train), y_train)

    grid = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_jobs=25, 
                            n_iter = 200, cv=3, scoring='f1', verbose=2)
    

    grid_result = grid.fit(X_train, y_train, class_weight=weights)

    #summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


if __name__ == '__main__':

    seed = 7
    np.random.seed(seed)

    X, y = create_data()
    print('Ratio true pulsars to false pulsars:', np.sum(y)/(len(y)-np.sum(y)))
    print('Ratio true pulsars to total datapoints:', np.sum(y)/len(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=seed)
    X_train, X_test = tools.scale_data(X_train, X_test, cat_split=None)



    X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=True, stratify=y_train)

    print('Ratio pulsars to false pulsars:')
    print('   test set:      ', np.sum(y_test)/(len(y_test)-np.sum(y_test)))
    print('   validation set:', np.sum(y_val)/(len(y_val)-np.sum(y_val)))
    print('   train set:     ', np.sum(y_train)/(len(y_train)-np.sum(y_train)))


    #nn_tuning(X_train, y_train, resample=0, use_weights=0)
    #nn_random_tuning(X_train, y_train, resample=0, use_weights=0)

    fit_NN(X_train2, X_val, X_test, y_train2, y_val, y_test, resample=0, use_weights=0, epochs=400, batch_size=64, optimizer='Nadam', n1=10, n2=None, dropout_rate=0.2)
