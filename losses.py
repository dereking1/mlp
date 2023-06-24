import numpy as np


# TODO: Consider Loss Class
def negative_log_likelihood(predicted, actual):
    samples = len(actual)
    correct_logprobs = -np.log(predicted[range(samples), actual])
    data_loss = np.sum(correct_logprobs) / samples
    return data_loss


def nll_derivative(predicted, actual):
    num_samples = len(actual)
    ## compute the gradient on predictions
    dscores = predicted
    dscores[range(num_samples), actual] -= 1
    dscores /= num_samples
    return dscores


def hinge(predicted, actual):
    # TODO Part 5
    N = len(actual)
    correct = predicted[range(N), actual]
    margins = np.maximum(0, 1 + predicted - correct.reshape(-1, 1))
    margins[range(N), actual] = 0
    return np.sum(margins) / N

def hinge_derivative(predicted, actual):
    # TODO Part 5
    N = len(actual)
    grad = np.zeros_like(predicted)
    correct = predicted[range(N), actual]
    margins = np.maximum(0, 1 + predicted - correct.reshape(-1, 1))
    margins[range(N), actual] = 0
    margins[margins > 0] = 1
    row_sums = np.sum(margins, axis=1)
    grad[range(N), actual] -= row_sums
    return grad / N


def mse(predicted, actual):
    # TODO Part 5
    # Hint: Convert actual to one-hot encoding
    dims = predicted.shape[1]
    actual1 = np.eye(dims)[actual]
    squared_errors = (predicted - actual1)**2
    row_losses = np.mean(squared_errors, axis=1)
    return np.mean(row_losses)


def mse_derivative(predicted, actual):
    # TODO Part 5
    N = len(actual)
    dims = predicted.shape[1]
    actual_onehot = np.eye(dims)[actual]
    return 2*(predicted - actual_onehot) / N

