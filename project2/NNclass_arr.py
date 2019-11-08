import numpy as np
np.random.seed(3155)

class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            n_hidden_neurons=[50],
            n_categories=10,
            epochs=10,
            batch_size=100,
            eta=0.1,
            lmbd=0.0,
            init_method='random',
            out_activation='softmax',
            cost_f='sigmoid'):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_layers = len(n_hidden_neurons)
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd

        self.init_method = init_method
        self.out_activation_key = out_activation # the type of acitvation function
        self.cost_f_key = cost_f # the type of cost function

        self.create_biases_and_weights()
        self.output_activation() # get correct output activation function
        self.cost_function() # get the correct cost function and its derivative

    def create_biases_and_weights(self):
        #self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        #self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.sizes = [self.n_features] + self.n_hidden_neurons# + [self.n_categories]
        self.hidden_weights = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        self.hidden_bias = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        
        if self.init_method == 'random':
            # layer 0=input layer
            for i in range(self.n_hidden_layers):
                self.hidden_weights[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])
                #self.hidden_bias[i+1] = np.zeros((1, self.sizes[i+1])) + 0.01
                self.hidden_bias[i+1] = np.zeros(self.sizes[i+1]) + 0.01
            
            self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_categories)
            self.output_bias = np.zeros(self.n_categories) + 0.01

        elif self.init_method == 'Xavier':
            # layer 0=input layer
            for i in range(self.n_hidden_layers):
                self.hidden_weights[i+1] = np.random.randn(self.sizes[i], self.sizes[i+1])*np.sqrt(1/self.sizes[i])
                #self.hidden_bias[i+1] = np.zeros((1, self.sizes[i+1])) + 0.01
                self.hidden_bias[i+1] = np.random.randn(1,self.sizes[i+1])
            
            self.output_weights = np.random.randn(self.n_hidden_neurons[-1], self.n_categories)*np.sqrt(1/self.sizes[-1])
            self.output_bias = np.random.randn(1, self.n_categories)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def grad_sigmoid(self, s):
        return s*(1-s)

    def linear(self,x):
        return x

    def grad_linear(self,x):
        return np.ones_like(x)

    def softmax(self,x):
        exp_term = np.exp(x)
        return exp_term/np.sum(exp_term, axis=1, keepdims=True)

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((self.Y_data - y_pred)**2)
    
    def grad_mean_squared_error(self, x):
        # DOUBLE CHECK THIS
        dcost = x -self.Y_data # *-1?
        dpred = a_o_deriv(x)
        return dcost*dpred
        

    def output_activation(self):
        # fix this, here and in outputlayer: DONE
        a_o_options = {
                'linear': self.linear,   # regression
                'sigmoid': self.sigmoid, # binary (one class) classification
                'softmax': self.softmax  # multiclass classification
                }
        self.a_o_f = a_o_options[self.out_activation_key]
        

    def cost_function(self):
        # should define cost function and its derivative

        cost_f_options = {
                'mse': [self.mean_squared_error, self.grad_mean_squared_error], # regression
                'sigmoid': [self.sigmoid, self.grad_sigmoid] # classification
                }
        self.cost, self.cost_deriv = cost_f_options[self.cost_f_key]


    def feed_forward(self):
        # feed-forward for training

        self.z_h = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        self.a_h = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        self.a_h[0] = self.X_data#.dot(self.hidden_weights[0]) + self.hidden_bias[0] # x.reshape(1, -1)
        for i in range(self.n_hidden_layers):
            self.z_h[i+1] = self.a_h[i].dot(self.hidden_weights[i+1]) + self.hidden_bias[i+1]
            self.a_h[i+1] = self.sigmoid(self.z_h[i+1])

        self.z_o = np.matmul(self.a_h[self.n_hidden_layers], self.output_weights) + self.output_bias

        #exp_term = np.exp(self.z_o)
        #self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        self.a_o = self.a_o_f(self.z_o)
        #self.a_h[self.n_hidden_layers+1]

    def feed_forward_out(self, X):
        # feed-forward for output
        #z_h = X.dot(self.hidden_weights) + self.hidden_bias
        #a_h = self.sigmoid(z_h)

        z_h = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        a_h = np.zeros(self.n_hidden_layers+1, dtype=np.ndarray)
        a_h[0] = X
        for i in range(self.n_hidden_layers):
            
            z_h[i+1] = a_h[i].dot(self.hidden_weights[i+1]) + self.hidden_bias[i+1]
            a_h[i+1] = self.sigmoid(z_h[i+1])
        
        z_o = np.matmul(a_h[self.n_hidden_layers], self.output_weights) + self.output_bias

        #exp_term = np.exp(z_o)
        #probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True) #softmax
        a_o = self.a_o_f(z_o)
        return a_o# probabilities

    def backpropagation(self):

        nh = self.n_hidden_layers
        # --------- dz[nh+1] = self.a[nh+1] - y = probabilities - Y.data
        #error_output = self.probabilities - self.Y_data
        error_output = self.a_o - self.Y_data # 
        # ------------- dz[nh] = da.T*sigmoid_grad(a_h[nh])
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.cost_deriv(self.a_h[nh])#self.a_h[nh] * (1 - self.a_h[nh])

        # ---------- dW[nhs+1] = a_h[nh].T.dot(dz[nh+1])
        self.output_weights_gradient = np.matmul(self.a_h[nh].T, error_output)
        # -------------- dB[nh+1] = np.sum(dz[nh+1], axis=0)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.zeros(nh+1, dtype=np.ndarray) # dw_h
        self.hidden_bias_gradient = np.zeros(nh+1, dtype=np.ndarray)  # db_h
        self.dz_h = np.zeros(nh+1, dtype=np.ndarray)
        self.da_h = np.zeros(nh+1, dtype=np.ndarray)
        
        #self.dz[nh+1] = error_output
        self.dz_h[nh] = error_hidden

        #self.da[nh+1] = self.dz[nh+1].dot(self.output_weights.T)
        
        #---------- dW[nh] = a_h[hidden_layers-1].T.dot(dz[nh])
        self.hidden_weights_gradient[nh] = self.a_h[nh].T.dot(error_hidden)
        #----------- dB[nh] = np.sum(dz[nh], axis=0)
        self.hidden_bias_gradient[nh] = np.sum(error_hidden, axis=0)
        #self.da_h[nh] = self.dz_h[nh+1].dot(self.hidden_weights[nh+1].T)
        #self.da_h[nh] = error_output.dot(self.output_weights.T)
        
        for i in range(nh, 0, -1):
            self.hidden_weights_gradient[i] = self.a_h[i-1].T.dot(self.dz_h[i])
            self.hidden_bias_gradient[i] = np.sum(self.dz_h[i], axis=0)
            self.da_h[i-1] = self.dz_h[i].dot(self.hidden_weights[i].T)
            #self.dz_h[i-1] = self.da_h[i-1]*(self.a_h[i-1]*(1-self.a_h[i-1]))
            self.dz_h[i-1] = self.da_h[i-1]*self.cost_deriv(self.a_h[i-1])
            # DONE: change sigmoid when doing lin. reg in dz_h
        
        #print('dz grad', self.dz_h)


        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient = self.lmbd * self.hidden_weights
        
        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient


    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()