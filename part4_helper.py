import plotting
import datasets
import mlp
import matplotlib.pyplot as plt
import numpy as np
import os


# This helper function will be useful for Parts 4B - 4D
def run_experiments(A_dataset_points, B_dataset_points, C_dataset_points, changing_var, changing_type, plot_type='log', save=False):
    title = f"{changing_type} - Train vs. Test Error"
    f, all_axs = plt.subplots(1, 3, figsize=(30, 10))
    if changing_type == 'epochs':
        g, grad_axs = plt.subplots(1, 3, figsize=(30, 10))
    else:
        grad_axs = [None, None, None]

    A_mlp = mlp.MLP([6, 4, 2])
    B_mlp = mlp.MLP([6, 4, 2])
    C_mlp = mlp.MLP([6, 4, 3])

    all_mlps = [A_mlp, B_mlp, C_mlp]
    names = ["Gaussian", "Circles", "Iris"]
    all_datasets = [A_dataset_points, B_dataset_points, C_dataset_points]

    # TODO: Fill in optimal learning rates after part4 
    learning_rates = [1e-3, 1e-2, 1e-2]

    for model, ax, name, data, lr, g_ax in zip(all_mlps, all_axs, names, all_datasets, learning_rates, grad_axs):
        # TODO: Part 4 A train-test split
        X_train, X_test, y_train, y_test = datasets.train_test_split_(data)

        train_err, test_err, model = plotting.get_errors(model,
                                                         X_train,
                                                         y_train,
                                                         X_test,
                                                         y_test,
                                                         lr,
                                                         changing_var,
                                                         changing_type)

        if plot_type == 'log':
            ax = plotting.plot_error_xlogscale(train_err, test_err, changing_var, changing_type, ax)
        else:
            ax = plotting.plot_error(train_err, test_err, changing_var, changing_type, ax)
        ax.set_title(name)

        if changing_type == "epochs":
            # TODO: Part 4 C iii plot gradients
            grads = [gr['dW'] for gr in model.gradients]
            grad_norms = [np.linalg.norm(gr) for gr in grads]
            g_ax = plotting.plot_gradients(grad_norms, g_ax)
            g_ax.set_title(name)

    f.suptitle(title)
    if(save):
        os.makedirs('figs', exist_ok=True)
        f.savefig(f'figs/{changing_type}')
    else:
        plt.show()

    if changing_type == "epochs":
        g.suptitle('gradients')
        g.savefig(f'figs/gradients')

# This helper function will be useful for Parts 4E
def get_errors_simple(model, X_train, y_train, X_test, y_test, lr=1e-3):
    # defaults if unchanged
    epochs = 1000

    train_err = np.zeros((3))
    test_err = np.zeros((3))

    for i in range(1):
        model.train(X_train, y_train, epochs, lr)
        try:
            train_err[i] = model.get_losses()[-1]
        except:
            train_err[i] = None
        model.predict(X_test, y_test)
        try:
            test_err[i] = model.test_loss
        except:
            test_err[i] = None

    return train_err, test_err


# This helper function will be useful for Parts 4F
def run_layer_experiments(A_dataset_points, B_dataset_points, C_dataset_points, layer_number, layer_multiplier, save=True):
    title = f"Layers - Train vs. Test Error"
    f, all_axs = plt.subplots(1, 3, figsize=(30, 10))

    wide_layer_num = 4
    assert (layer_number > wide_layer_num)
    base_number = 4
    thin_layers = []
    wide_layers = []
    for l in range(layer_number):
        thin_layers.append(base_number)
    for l in range(wide_layer_num):
        wide_layers.append(base_number * layer_multiplier)

    binary_thin_layers = thin_layers.copy()
    binary_thin_layers.append(2)
    binary_wide_layers = wide_layers.copy()
    binary_wide_layers.append(2)
    trinary_thin_layers = thin_layers.copy()
    trinary_thin_layers.append(3)
    trinary_wide_layers = wide_layers.copy()
    trinary_wide_layers.append(3)

    A_mlp_thin = mlp.MLP(binary_thin_layers)
    B_mlp_thin = mlp.MLP(binary_thin_layers)
    C_mlp_thin = mlp.MLP(trinary_thin_layers)

    A_mlp_wide = mlp.MLP(binary_wide_layers)
    B_mlp_wide = mlp.MLP(binary_wide_layers)
    C_mlp_wide = mlp.MLP(trinary_wide_layers)

    # TODO: Compare each thin dataset with the wide dataset
    thin_mlps = [A_mlp_thin, B_mlp_thin, C_mlp_thin]
    wide_mlps = [A_mlp_wide, B_mlp_wide, C_mlp_wide]
    names = ["Gaussian", "Circles", "Iris"]
    all_datasets = [A_dataset_points, B_dataset_points, C_dataset_points]

    # TODO: Fill in optimal learning rates after part4 
    learning_rates = [1e-4, 1e-4, 1e-4]
    learning_rates = [1e-3, 1e-2, 1e-2]

    for thin_model, wide_model, ax, name, data, lr in zip(thin_mlps, wide_mlps, all_axs, names, all_datasets,
                                                          learning_rates):
        X_train, X_test, y_train, y_test = datasets.train_test_split_(data)
        thin_train_err, thin_test_err = get_errors_simple(thin_model, X_train, y_train, X_test, y_test, lr=lr)
        wide_train_err, wide_test_err = get_errors_simple(wide_model, X_train, y_train, X_test, y_test, lr=lr)
        
        ax = plotting.plot_layer_size_error(thin_train_err, thin_test_err, 'thin', ax)
        ax = plotting.plot_layer_size_error(wide_train_err, wide_test_err, 'wide', ax)
        ax.set_title(name)

    f.suptitle(title)
    # plt.show()
    if (save):
        os.makedirs('figs', exist_ok=True)
        plt.savefig(f'figs/layer_sizes')
    else:
        plt.show()
