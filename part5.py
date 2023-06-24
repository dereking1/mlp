import datasets
import part4_helper

N = 100
M = 100

dims = 3
A_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kGaussian, dims).get_dataset()
B_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kCircles).get_dataset()
C_dataset_points = datasets.generate_nd_dataset(N, M, datasets.kIris).get_dataset()

activation_funcs = ['sigmoid', 'relu', 'tanH']
# TODO: see how various activation functions perform
part4_helper.run_experiments(A_dataset_points, B_dataset_points, C_dataset_points, activation_funcs, 'activation_func',
                             plot_type='linear', save=True)

losses = ['negative_log_likelihood', 'hinge', 'mse']
# TODO: see how various loss functions perform
part4_helper.run_experiments(A_dataset_points, B_dataset_points, C_dataset_points, losses, 'loss_func',
                             plot_type='linear', save=True)
