import numpy as np
from datasets import *
from random_control import *
from losses import *
from plotting import *


class Layer:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons

    def softmax(self, inputs):
        out = []
        for row in inputs:
            out.append(np.exp(row)/np.sum(np.exp(row)))
        return np.array(out)

    def tanH(self, inputs):
        return np.tanh(inputs)

    def sigmoid(self, inputs):
        out = None
        try:
            out = np.array([1/(1+np.exp(-x)) for x in inputs])
        except TypeError:
            out = 1/(1+np.exp(-inputs))
        return out

    def relu(self, inputs):
        out = np.zeros(inputs.shape)
        for i in range(inputs.shape[0]):
            if len(inputs.shape) == 2:
                for j in range(inputs.shape[1]):
                    out[i][j] = max(inputs[i][j], 0)
            else:
                out[i] = max(inputs[i], 0)
        return out

    def tanH_derivative(self, inputs):
        # TODO (Part 5)
        return 1-np.square(self.tanH(inputs))

    def sigmoid_derivative(self, inputs):
        return self.sigmoid(inputs)*(1-self.sigmoid(inputs))

    def relu_derivative(self, inputs):
        # TODO (Part 5)
        out = np.zeros(inputs.shape)
        for i in range(inputs.shape[0]):
            if len(inputs.shape) == 2:
                for j in range(inputs.shape[1]):
                    x = inputs[i][j]
                    out[i][j] = 1 if x > 0 else 0
            else:
                out[i] = 1 if inputs[i] > 0 else 0
        return out

    def apply_chain_rule_activation_derivative(self, Z, activation_derivative):
        # TODO rename the variable q appropriately -- what should this be? (q renamed to Z)
        if activation_derivative == 'relu':
            dZ = self.relu_derivative(inputs=Z)
        elif activation_derivative == 'sigmoid':
            dZ = self.sigmoid_derivative(inputs=Z)
        elif activation_derivative == 'tanH':
            dZ = self.tanH_derivative(inputs=Z)
        else:
            raise ValueError('Activation derivative not supported: ' + activation_derivative)
        return dZ

    def forward(self, inputs, weights, bias, activation):
        # TODO compute Z_curr from weights and bias
        Z_curr = np.dot(inputs, np.transpose(weights)) + bias

        if activation == 'relu':
            A_curr = self.relu(inputs=Z_curr)
        elif activation == 'sigmoid':
            A_curr = self.sigmoid(inputs=Z_curr)
        elif activation == 'tanH':
            A_curr = self.tanH(inputs=Z_curr)
        elif activation == 'softmax':
            A_curr = self.softmax(inputs=Z_curr)
        else:
            raise ValueError('Activation function not supported: ' + activation)

        return A_curr, Z_curr

    def backward(self, dA_curr, W_curr, Z_curr, A_prev, activation):

        # TODO each of these functions require you to compute all of the colored terms in the Part 2 Figure.
        # We will denote the partial derivative of the loss with respect to each variable as dZ, dW, db, dA
        # These variable map to the corresponding terms in the figure. Note that these are matrices and not individual
        # values, you will determine how to vectorize the code yourself. Think carefully about dimensions!
        # You can use the self.apply_chain_rule_activation_derivative() function, although there are solutions without it.
        '''
        The inputs to this function are:
            dA_curr - the partial derivative of the loss with respect to the activation of the preceding layer (l + 1).
            W_curr - the weights of the layer (l)
            Z_curr - the weighted sum of layer (l)
            A_prev - the activation of this layer (l) ... we use prev with respect to dA_curr

        The outputs are the partial derivatives with respect
            dA - the activation of this layer (l) -- needed to continue the backprop
            dW - the weights -- needed to update the weights
            db - the bias -- (needed to update the bias
        '''

        if activation == 'softmax':
            # We deal with the softmax function for you, so dZ is not needed for this one. dA_curr = dZ for this one.
            dW = np.dot(np.transpose(A_prev), dA_curr)
            db = np.sum(dA_curr, axis=0)
            dA = np.dot(dA_curr, W_curr) 
        elif activation == 'sigmoid':
            # Computing dZ is not technically needed, but it can be used to help compute the other values.
            activation_derivative = 'sigmoid'
            dZ = dA_curr * self.apply_chain_rule_activation_derivative(Z_curr, activation_derivative)
            dW = np.dot(np.transpose(A_prev), dZ)
            db = np.sum(dZ, axis=0)
            dA = np.dot(dZ, W_curr) 
        elif activation == 'tanH':
            activation_derivative = 'tanH'
            dZ = dA_curr * self.apply_chain_rule_activation_derivative(Z_curr, activation_derivative)
            dW = np.dot(np.transpose(A_prev), dZ)
            db = np.sum(dZ, axis=0)
            dA = np.dot(dZ, W_curr) 
        elif activation == 'relu':
            activation_derivative = 'relu'
            dZ = dA_curr * self.apply_chain_rule_activation_derivative(Z_curr, activation_derivative)
            dW = np.dot(np.transpose(A_prev), dZ)
            db = np.sum(dZ, axis=0)
            dA = np.dot(dZ, W_curr) 
        else:
            raise ValueError('Activation function not supported: ' + activation)

        return dA, dW, db

'''
* `MLP` is a class that represents the multi-layer perceptron with a variable number of hidden layer. 
   The constructor initializes the weights and biases for the hidden and output layers.
* `sigmoid`, `relu`, `tanh`, and `softmax` are activation function used in the MLP. 
   They should each map any real value to a value between 0 and 1.
* `forward` computes the forward pass of the MLP. 
   It takes an input X and returns the output of the MLP.
* `sigmoid_derivative`, `relu_derivative`, `tanH_derivative` are the derivatives of the activation functions. 
   They are used in the backpropagation algorithm to compute the gradients.
*  `mse_loss`, `hinge_loss`, `cross_entropy_loss` are each loss functions.
   The MLP algorithms optimizes to minimize those.
* `backward` computes the backward pass of the MLP. It takes the input X, the true labels y, 
   the predicted labels y_hat, and the learning rate as inputs. 
   It computes the gradients and updates the weights and biases of the MLP.
* `train` trains the MLP on the input X and true labels y. It takes the number of epochs 
'''

class MLP:
    def __init__(self, layer_list):
        '''
        Arguments
        --------------------------------------------------------
        layer_list: a list of numbers that specify the width of the hidden layers. 
               The dataset dimensionality (input layer) and output layer (1) 
               should not be specified.
        '''
        self.layer_list = layer_list
        self.network = []  ## layers
        self.architecture = []  ## mapping input neurons --> output neurons
        self.params = []  ## W, b
        self.memory = []  ## Z, A
        self.gradients = []  ## dW, db
        self.loss = []
        self.accuracy = []

        self.loss_func = None
        self.loss_derivative = None

        self.init_from_layer_list(self.layer_list)

    # TODO read and understand the next several functions, you will need to understand them to complete the assignment.
    #  In particular, you will need to understand
    #  self.network, self.architecture, self.params, self.memory, and self.gradients. It may be helpful to write some
    #  notes about what each of these variables are and how they are used.
    def init_from_layer_list(self, layer_list):
        for layer_size in layer_list:
            self.add(Layer(layer_size))

    def add(self, layer):
        self.network.append(layer)

    def _compile(self, data, activation_func='relu'):
        self.architecture = [] 
        for idx, layer in enumerate(self.network):
            if idx == 0:
                self.architecture.append({'input_dim': data.shape[1], 'output_dim': self.network[idx].num_neurons,
                                          'activation': activation_func})
            elif idx > 0 and idx < len(self.network) - 1:
                self.architecture.append(
                    {'input_dim': self.network[idx - 1].num_neurons, 'output_dim': self.network[idx].num_neurons,
                     'activation': activation_func})
            else:
                self.architecture.append(
                    {'input_dim': self.network[idx - 1].num_neurons, 'output_dim': self.network[idx].num_neurons,
                     'activation': 'softmax'})
        return self

    def _init_weights(self, data, activation_func, seed=None):
        self.params = []
        self._compile(data, activation_func)

        if seed is None:
            for i in range(len(self.architecture)):
                self.params.append({
                    'W': generator.uniform(low=-1, high=1,
                                           size=(self.architecture[i]['output_dim'],
                                                 self.architecture[i]['input_dim'])),
                    'b': np.zeros((1, self.architecture[i]['output_dim']))})
        else:
            # For testing purposes
            fixed_generator = np.random.default_rng(seed=seed)
            for i in range(len(self.architecture)):
                self.params.append({
                    'W': fixed_generator.uniform(low=-1, high=1,
                                           size=(self.architecture[i]['output_dim'],
                                                 self.architecture[i]['input_dim'])),
                    'b': np.zeros((1, self.architecture[i]['output_dim']))})

        return self

    def forward(self, data):
        A_prev = data
        A_curr = None
        self.memory = [{} for _ in range(len(self.params))]

        for i in range(len(self.params)):

            # TODO compute the forward for each layer and store the appropriate values in the memory.
            # We format our memory_list as a list of dicts, please follow this format.
            # mem_dict = {'?': ?}; self.memory.append(mem_dict)
            weights, bias = self.params[i]['W'], self.params[i]['b']
            layer = self.network[i]
            activation_func = self.architecture[i]['activation']
            A_curr, Z_curr = layer.forward(A_prev, weights, bias, activation_func)
            self.memory[i]['A'] = A_prev  
            self.memory[i]['Z'] = Z_curr
            A_prev = A_curr

        return A_curr

    def backward(self, predicted, actual):
        ## compute the gradient on predictions
        dscores = self.loss_derivative(predicted, actual)
        dA_prev = dscores  # This is the derivative of the loss function with respect to the output of the last layer

        # TODO compute the backward for each layer and store the appropriate values in the gradients.
        # We format our gradients_list as a list of dicts, please follow this format (same as self.memory).

        self.gradients = [{} for _ in range(len(self.params))]
        for i in range(len(self.params)-1, -1, -1):
            W_curr = self.params[i]['W']
            Z_curr, A_prev = self.memory[i]['Z'], self.memory[i]['A']
            layer = self.network[i]
            activation_func = self.architecture[i]['activation']
            dA, dW, db = layer.backward(dA_prev, W_curr, Z_curr, A_prev, activation_func)
            self.gradients[len(self.params) - i - 1]['dW'] = dW
            self.gradients[len(self.params) - i - 1]['db'] = db
            dA_prev = dA

    def _update(self, lr):
        # TODO update the network parameters using the gradients and the learning rate.
        #  Recall gradients is a list of dicts, and params is a list of dicts, pay attention to the order of the dicts.
        #  Is gradients the same order as params? This might depend on your implementations of forward and backward.
        #  Should we add or subtract the deltas?
        for i in range(len(self.params)):
            dW, db = tuple(self.gradients[len(self.params) - i - 1].values())
            self.params[i]['W'] = self.params[i]['W'] - lr * np.transpose(dW)
            self.params[i]['b'] = self.params[i]['b'] - lr * db
    
    # Loss and accuracy functions
    def _calculate_accuracy(self, predicted, actual):
        return np.mean(np.argmax(predicted, axis=1) == actual)

    def _calculate_loss(self, predicted, actual):
        return self.loss_func(predicted, actual)

    def _set_loss_function(self, loss_func_name='negative_log_likelihood'):
        if loss_func_name == 'negative_log_likelihood':
            self.loss_func = negative_log_likelihood
            self.loss_derivative = nll_derivative
        elif loss_func_name == 'hinge':
            self.loss_func = hinge
            self.loss_derivative = hinge_derivative
        elif loss_func_name == 'mse':
            self.loss_func = mse
            self.loss_derivative = mse_derivative
        else:
            raise Exception("Loss has not been specified. Abort")

    def get_losses(self):
        if len(self.loss) > 0:
            return self.loss
        else:
            return [np.inf]

    def get_accuracy(self):
        if len(self.accuracy) > 0:
            return self.accuracy
        else:
            return [0]

    def train(self, X_train, y_train, epochs=1000, lr=1e-4, batch_size=16, activation_func='relu', loss_func='negative_log_likelihood'):

        self.loss = []
        self.accuracy = []

        # cast to int
        y_train = y_train.astype(int)

        # initialize network weights
        self._init_weights(X_train, activation_func)

        # TODO calculate number of batches
        num_datapoints = len(y_train)
        num_batches = num_datapoints // batch_size
        if num_datapoints % batch_size != 0:
            num_batches += 1

        # TODO shuffle the data and iterate over mini-batches for each epoch.
        #  We are implementing mini-batch gradient descent.
        #  How you batch the data is up to you, but you should remember shuffling has to happen the same way for
        #  both X and y.
        self._set_loss_function(loss_func_name=loss_func)

        for i in range(int(epochs)):

            batches = []
            idx = list(range(num_datapoints))
            random.shuffle(idx)
            k = 0
            for j in range(num_batches-1):
                X_batch = []
                y_batch = []
                for _ in range(j * batch_size,(j + 1)*batch_size):
                    X_batch.append(X_train[idx[k]])
                    y_batch.append(y_train[idx[k]])
                    k += 1
                batches.append((np.array(X_batch), np.array(y_batch).astype(int)))
            X_batch = []
            y_batch = []
            while k < len(idx):
                X_batch.append(X_train[idx[k]])
                y_batch.append(y_train[idx[k]])
                k += 1
            batches.append((np.array(X_batch), np.array(y_batch).astype(int)))

            batch_loss = 0
            batch_acc = 0
            for batch_num in range(num_batches):
                X_batch, y_batch = batches[batch_num]

                # TODO Hint: do any variables need to be reset each pass?
                # Gradients reset in MLP backward function
                yhat = self.forward(X_batch) # TODO compute yhat
                acc = self._calculate_accuracy(yhat, y_batch)  # TODO compute and update batch acc
                loss = self._calculate_loss(yhat, y_batch)  # TODO compute and update batch loss

                # Stop training if loss is NaN, why might the loss become NaN or inf?
                if np.isnan(loss) or np.isinf(loss):
                    if len(self.accuracy) == 0:
                        s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, acc, loss)
                    else:
                        s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, self.accuracy[-1], self.loss[-1])
                    print(s)
                    print("Stopping training because loss is NaN")
                    return

                # TODO update the network
                batch_loss += loss
                batch_acc += acc
                self.backward(yhat, y_batch)
                self._update(lr=lr)

            self.loss.append(batch_loss / num_batches)
            self.accuracy.append(batch_acc / num_batches)

            if i % 20 == 0:
                s = 'EPOCH: {}, LR: {}, ACCURACY: {}, LOSS: {}'.format(i, lr, self.accuracy[-1], self.loss[-1])
                print(s)

    def predict(self, X, y, loss_func='negative_log_likelihood'):
        # TODO predict the loss and accuracy on a val or test set and print the results. Make sure to gracefully handle
        #  the case where the loss is NaN or inf.
        self._set_loss_function(loss_func_name=loss_func)
        yhat = self.forward(X)
        acc = self._calculate_accuracy(yhat, y)
        loss = self._calculate_loss(yhat, y)

        # for plotting purposes
        self.test_loss = loss  # TODO loss
        self.test_accuracy = acc  # TODO accuracy


if __name__ == '__main__':
    # Copy of part2.py in case useful for debugging
    N = 100
    M = 100
    dims = 3
    gaus_dataset_points = generate_nd_dataset(N, M, kGaussian, dims).get_dataset()
    X = gaus_dataset_points[:, :-1]
    y = gaus_dataset_points[:, -1].astype(int)

    model = MLP([3,2])
    model.train(X, y)
    plot_losses(title='Training Loss Plot', losses=model.get_losses(), save=True)
    plot_accuracies(title='Training Accuracy', accuracies=model.get_accuracy(), save=True)