from random_control import *
from mlp import MLP, Layer
import pickle as pkl
import os
import sys
import numpy as np



'''
You can run this file to test your functions.
You should not need to modify this file, but you can if you want to get more specifics on errors.

To run this file, you should have a folder called seed=0_correct_function_output, this folder
should contain the correct outputs for each function. 

We purposefully leave off tests for train() and update(). If you are successfully passing the tests here,
you should be able to diagnose any issues with train() and update() on your own by looking at the plots.

In theory you may pass a test with incorrect code although this is very unlikely.

When we grade, we will run these tests with a different seed, so you should not hard code any values.
'''

seed = 0  # do not change this, and also no need to change the seed in random control.
output_folder = './seed=0_correct_function_output/'
input_size = 100


def test_activation(activation_func='sigmoid'):
    np.random.seed(seed)

    dl = Layer(1)
    inputs = np.random.random(size=input_size)
    if activation_func == 'sigmoid':
        outputs = dl.sigmoid(inputs)
    elif activation_func == 'tanH':
        outputs = dl.tanH(inputs)
    elif activation_func == 'relu':
        outputs = dl.relu(inputs)
    else:
        raise ValueError('unknown activation func')

    with open(os.path.join(output_folder, f'{activation_func}.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)

    assert np.allclose(outputs, corr_outputs), f'{activation_func} failed -- outputs are not close enough'


def test_activation_derivative(activation_func='sigmoid'):
    np.random.seed(seed)

    dl = Layer(1)
    inputs_Z = np.random.random(size=input_size)
    if activation_func == 'sigmoid':
        outputs = dl.sigmoid_derivative(inputs_Z)
    elif activation_func == 'tanH':
        outputs = dl.tanH_derivative(inputs_Z)
    elif activation_func == 'relu':
        outputs = dl.relu_derivative(inputs_Z)
    else:
        raise ValueError('unknown activation func')
    with open(os.path.join(output_folder, f'{activation_func}_derivative.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)

    assert np.allclose(outputs, corr_outputs), f'{activation_func} failed -- outputs are not close enough'


def test_softmax():
    np.random.seed(seed)

    dl = Layer(1)
    num_out = 2
    inputs = np.random.random(size=(input_size, num_out))
    outputs = dl.softmax(inputs)
    with open(os.path.join(output_folder, f'softmax.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)

    assert np.allclose(outputs, corr_outputs), f'softmax function failed -- outputs are not close enough'


def test_forward():
    np.random.seed(seed)

    num_neurons = 6
    inputs = np.random.random((input_size, 3))
    weights = np.random.random((num_neurons, 3))
    bias = np.random.random((1, num_neurons))
    dl = Layer(num_neurons)
    sig_out = dl.forward(inputs, weights, bias, activation='sigmoid')
    relu_out = dl.forward(inputs, weights, bias, activation='relu')
    tan_out = dl.forward(inputs, weights, bias, activation='tanH')
    softmax_out = dl.forward(inputs, weights, bias, activation='softmax')

    outputs = dict(sigmoid_out=sig_out,  relu_out=relu_out, tanH_out=tan_out, softmax_out=softmax_out)
    with open(os.path.join(output_folder, f'forward_out.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)

    for key in outputs.keys():
        assert np.allclose(outputs[key], corr_outputs[key]), f'{key} failed -- outputs are not close enough'


def test_backward():
    np.random.seed(seed)

    num_neurons = 10
    num_out = 2
    dA_curr = np.random.random((input_size, num_out))
    W_curr = np.random.random((num_out, num_neurons))
    Z_curr = np.random.random((input_size, num_out))
    A_prev = np.random.random((input_size, num_neurons))
    dl = Layer(num_neurons)

    sig_out = dl.backward(dA_curr, W_curr, Z_curr, A_prev, activation='sigmoid')
    relu_out = dl.backward(dA_curr, W_curr, Z_curr, A_prev, activation='relu')
    tan_out = dl.backward(dA_curr, W_curr, Z_curr, A_prev, activation='tanH')
    softmax_out = dl.backward(dA_curr, W_curr, Z_curr, A_prev, activation='softmax')

    outputs = dict(sigmoid_out=sig_out,  relu_out=relu_out, tanH_out=tan_out, softmax_out=softmax_out)
    with open(os.path.join(output_folder, f'backward_out.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)

    for key in outputs.keys():
        for i in range(len(outputs[key])):
            assert np.allclose(outputs[key][i], corr_outputs[key][i]), f'{key} failed -- outputs are not close enough'


def test_mlp_forward():
    np.random.seed(0)

    dims = 3
    out_size = 2
    model = MLP([6, 8, 10, out_size])
    model._set_loss_function('negative_log_likelihood')
    inputs = np.random.random((input_size, dims))
    model._init_weights(inputs, 'relu', seed=seed)
    yhat = model.forward(inputs)
    y_train = np.random.randint(0, 2, size=(input_size))
    acc = model._calculate_accuracy(predicted=yhat, actual=y_train)
    loss = model._calculate_loss(predicted=yhat, actual=y_train)

    outputs = dict(yhat=yhat,  y_train=y_train, acc=acc, loss=loss)
    with open(os.path.join(output_folder, f'mlp_forward.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)

    assert np.allclose(outputs['yhat'], corr_outputs['yhat']), f'forward failed -- yhat outputs are not close enough'
    assert np.allclose(outputs['y_train'], corr_outputs['y_train']), f'forward failed -- y_train outputs are not close enough'
    assert np.allclose(outputs['acc'], corr_outputs['acc']), f'forward failed -- acc outputs are not close enough'
    assert np.allclose(outputs['loss'], corr_outputs['loss']), f'forward failed -- loss outputs are not close enough'


def test_mlp_backward():
    np.random.seed(0)

    dims = 3
    out_size = 2
    model = MLP([6, 8, 10, out_size])
    model._set_loss_function('negative_log_likelihood')
    inputs = np.random.random((input_size, dims))
    model._init_weights(inputs, 'relu', seed=seed)
    yhat = model.forward(inputs)
    inputs_predicted = np.random.random((input_size, out_size))
    inputs_actual = np.random.randint(0, 2, size=(input_size))
    model._set_loss_function('negative_log_likelihood')
    model.backward(inputs_predicted, inputs_actual)

    outputs = dict(yhat=yhat, gradients=model.gradients)
    with open(os.path.join(output_folder, f'mlp_backward.pkl'), 'rb') as f:
        corr_outputs = pkl.load(f)

    assert np.allclose(outputs['yhat'], corr_outputs['yhat']), f'forward failed -- yhat outputs are not close enough'
    for i in range(len(outputs['gradients'])):
        assert np.allclose(outputs['gradients'][i]['dW'], corr_outputs['gradients'][i]['dW']), f'forward failed -- gradient W outputs are not close enough'
        assert np.allclose(outputs['gradients'][i]['db'], corr_outputs['gradients'][i]['db']), f'forward failed -- gradient b outputs are not close enough'

def part2_tests():
    print(f"{'-'*5} starting {sys._getframe().f_code.co_name} {'-'*5}")
    test_activation('sigmoid')
    test_activation_derivative('sigmoid')
    test_softmax()    
    test_forward()
    test_backward()
    test_mlp_forward()
    test_mlp_backward()
    print(f"pass")


def part5_tests():
    print(f"{'-'*5} starting {sys._getframe().f_code.co_name} {'-'*5}")
    test_activation('relu')
    test_activation('tanH')

    test_activation_derivative('relu')
    test_activation_derivative('tanH')
    print(f"pass")


if __name__ == "__main__":
    part2_tests()
    part5_tests()
    print('All tests passed!')