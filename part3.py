import datasets
import mlp
import plotting


########################### A. Circles ##############################
N = 100
M = 100
dims = 2
dataset = datasets.generate_nd_dataset(N, M, datasets.kCircles)
dataset.save_dataset_plot()

circle_dataset_points = dataset.get_dataset()

X = circle_dataset_points[:, :-1]
y = circle_dataset_points[:, -1].astype(int)

model = mlp.MLP([dims, 2])

model.train(X, y, epochs=1000, lr=1e-4)
title = f'losses_circles_{dims}d_mlp'
plotting.plot_losses(title, model.get_losses(), save=True)
title = f'accuracies_circles_{dims}d_mlp'
plotting.plot_accuracies(title, model.get_accuracy(), save=True)


########################### B. Iris ##############################
dataset = datasets.generate_nd_dataset(N, M, datasets.kIris)

iris_dataset_points = dataset.get_dataset()

X = iris_dataset_points[:, :-1]
y = iris_dataset_points[:, -1].astype(int)
dims = X.shape[1]
num_classes = 3
model = mlp.MLP([dims, num_classes])

model.train(X, y, epochs=1000, lr=1e-4)
title = f'losses_iris_mlp'
plotting.plot_losses(title, model.get_losses(), save=True)
title = 'accuracies_iris_mlp'
plotting.plot_accuracies(title, model.get_accuracy(), save=True)
