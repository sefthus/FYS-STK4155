
def kfold_CV_old(X, z, z_true, return_var = False, d=None, lmbda=0):
    """ k-fold cross-validation without sklearn Kfold """
    """ make prediction based on noisy data z, but compare (MSE)"""
    """ with the true function z_true """
    
    splits = 5

    Xs, zs, zs_true = shuffle(X, z, z_true)#, random_state=0) # in consistent way
    
    Xk = np.array_split(Xs, splits)
    zk = np.array_split(zs, splits)
    z_truek = np.array_split(zs_true, splits)

    mse_splits_test = np.zeros(splits)
    mse_splits_train = np.zeros(splits)
    bias_splits = np.zeros(splits)
    variance_splits = np.zeros(splits)

    for i in range(int(splits)):
        
        X_test = Xk[i]
        z_test = zk[i]
        z_true_test = z_truek[i]

        X_train = np.concatenate(np.delete(Xk, i, 0))
        z_train = np.ravel(np.delete(zk, i, 0))
        z_true_train = np.ravel(np.delete(z_truek, i, 0))


        # fit model on training set
        Idm = np.identity(np.shape(X.T)[0])
        XtXinv_train = invert_matrix(np.matmul(X_train.T, X_train) + lmbda*Idm)

        beta = XtXinv_train.dot(X_train.T).dot(z_train)
        
        # evaulate model on test set
        z_tilde_test = X_test.dot(beta)
        z_tilde_train = X_train.dot(beta)


        mse_splits_test[i] = np.mean((z_test - z_tilde_test)**2)
        mse_splits_train[i] = np.mean((z_train - z_tilde_train)**2)
        bias_splits[i] = np.mean((z_true_test - np.mean(z_tilde_test))**2)
        variance_splits[i] = np.var(z_tilde_test)
    
    #print('beta=', beta)
    # calculate errors, and the bias and variance scores    
    mse_test = np.mean(mse_splits_test)
    mse_train = np.mean(mse_splits_train)
    bias = np.mean(bias_splits)
    variance = np.mean(variance_splits)
    print('error', mse_test)
    print('variance', variance)
    print('bias:', bias_splits)
    print('error: {} >= {}: variance + bias'.format(mse_test, bias+variance))

    if return_var:
        return mse_test, mse_train, bias, variance

    print('MSE_scores', mse_test)
