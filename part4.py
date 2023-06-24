import datasets
import mlp
import numpy as np
import plotting

import part4_helper


#### Generate datasets ####
N = 100
M = 100

dims = 3
A_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kGaussian, dims).get_dataset()
B_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kCircles).get_dataset()
C_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kIris).get_dataset()

# TODO: Learning rate experiments
learning_rates = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
part4_helper.run_experiments(A_dataset_points, B_dataset_points, C_dataset_points, learning_rates, "learning_rate", save=True)

# TODO: Epochs experiments
epochs = [1e1, 1e2, 1e3, 1e4]
part4_helper.run_experiments(A_dataset_points, B_dataset_points, C_dataset_points, epochs, "epochs", save=True)

# TODO: Batch size experiments
batch_sizes = [1, 5, 10, 25, 50]
part4_helper.run_experiments(A_dataset_points, B_dataset_points, C_dataset_points, batch_sizes, "batch_size", save=True)

# TODO Imabalanced datasets, varying sample sizes
print("Errors of imbalanced and of varying sample sets")
Ns = [50, 100, 1000, 1000]
Ms = [50, 100, 1000, 100]
gaus_train_errors = np.zeros((4, 3))
gaus_test_errors = np.zeros((4, 3))
circle_train_erorrs = np.zeros((4, 3))
circle_test_errors = np.zeros((4, 3))
i = 0
for n, m in zip(Ns, Ms):
    # TODO For the gaussian dataset, find errors of imbalanced datasets
    # The get_errors_of_imbalanced_sets function in `part4_helper.py` may assist
    gaus_dataset_points = datasets.generate_nd_dataset(n, m, datasets.kGaussian, 3).get_dataset()
    gaus_X_train, gaus_X_test, gaus_y_train, gaus_y_test = datasets.train_test_split_(gaus_dataset_points)
    gaus_mlp = mlp.MLP([6, 4, 2])
    gtr, gtst = part4_helper.get_errors_simple(gaus_mlp,
                                               gaus_X_train,
                                               gaus_y_train,
                                               gaus_X_test,
                                               gaus_y_test)
    gaus_train_errors[i, :] = gtr
    gaus_test_errors[i, :] = gtst

    # TODO For the circles dataset, find errors of imbalanced datasets
    circle_dataset_points = datasets.generate_nd_dataset(n, m, datasets.kCircles).get_dataset()
    circles_X_train, circles_X_test, circles_y_train, circles_y_test = datasets.train_test_split_(circle_dataset_points)
    circle_mlp = mlp.MLP([6, 4, 2])
    ctr, ctst = part4_helper.get_errors_simple(circle_mlp,
                                               circles_X_train,
                                               circles_y_train,
                                               circles_X_test,
                                               circles_y_test)
    circle_train_erorrs[i, :] = ctr
    circle_test_errors[i, :] = ctst
    i += 1

Ns[-1] = 10000
plotting.plot_training_differences(Ns, gaus_train_errors, gaus_test_errors, circle_train_erorrs, circle_test_errors,
                                   save=True)

# Layer number vs. layer width
layer_number = 15
layer_multiplier = 5
part4_helper.run_layer_experiments(A_dataset_points, B_dataset_points, C_dataset_points, layer_number, layer_multiplier,
                                   save=True)
