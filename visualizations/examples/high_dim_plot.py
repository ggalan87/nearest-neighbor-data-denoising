import sys
from pathlib import Path
import torch

sys.path.append('/home/amidemo/devel/workspace')

import visualizations.hd_decision_boundary.uci_loader as uci_loader
import numpy as np
from visualizations.hd_decision_boundary.decisionboundaryplot import DBPlot
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_digits, load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from numpy.random.mtrand import permutation
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA

from sklearn.manifold import TSNE
import umap
from evt.vast_openset import OpensetData, OpensetTrainer, OpensetModelParameters
from features_storage import FeaturesStorage


def demo():
    # Load data
    data = load_iris()

    # Obtain data, target
    X, y = data.data, data.target

    # The plot works only for binary classifiers. Below we set classes other than zero-0 to a single label
    y[y != 0] = 1

    # Shuffle the data
    random_idx = permutation(np.arange(len(y)))
    X = X[random_idx]
    y = y[random_idx]

    # Create model
    model = LogisticRegression(C=1)
    # model = RandomForestClassifier(n_estimators=10)

    # plot high-dimensional decision boundary
    db = DBPlot(model)
    db.fit(X, y, training_indices=None)
    db.plot(
        plt, generate_testpoints=False
    )  # set generate_testpoints=False to speed up plotting

    plt.show()

    # # plot learning curves for comparison
    # N = 10
    # train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    # plt.errorbar(
    #     train_sizes,
    #     np.mean(train_scores, axis=1),
    #     np.std(train_scores, axis=1) / np.sqrt(N),
    # )
    # plt.errorbar(
    #     train_sizes,
    #     np.mean(test_scores, axis=1),
    #     np.std(test_scores, axis=1) / np.sqrt(N),
    #     c="r",
    # )
    #
    # plt.legend(["Accuracies on training set", "Accuracies on test set"])
    # plt.xlabel("Number of data points")
    # plt.title(str(model))
    # plt.show()


def demo_evm():
    # Load data
    data = load_iris()

    # Obtain data, target
    X, y = data.data, data.target

    fs = FeaturesStorage('iris')
    data = \
        {
            'feats': X,
            'target': y
        }

    fs.add('trainval', data)
    training_feats = fs.training_feats
    o_data = OpensetData(features=training_feats['feats'], class_labels=training_feats['target'])

    approach = 'OpenMax'
    algorithm_parameters = "--distance_metric euclidean --tailsize 1.0"
    saver_parameters = f"--OOD_Algo {approach}"
    openset_model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

    model = OpensetTrainer(o_data, openset_model_params)
    model.train()
    # model = LogisticRegression(C=1)

    # del model._models['1']
    # del model._models['2']

    # plot high-dimensional decision boundary
    # mapper = umap.UMAP(verbose=True, densmap=False, n_epochs=30)
    # mapper = TSNE(n_components=2, init='random')
    mapper = PCA(n_components=2)

    #
    db = DBPlot(model, dimensionality_reduction=mapper, n_decision_boundary_keypoints=60)

    # The plot works only for binary classifiers. Below we set classes other than zero-0 to a single label
    y[y != 0] = 1

    # swap labels
    # y[y == 0] = 2
    # y[y == 1] = 0
    # y[y == 2] = 1

    db.fit(X, y, training_indices=None)

    db.plot(
        plt, plot_decision_points=False, generate_testpoints=False, generate_background=False
    )  # set generate_testpoints=False to speed up plotting

    plt.show()


def demo_heating():
    path = '/home/amidemo/devel/workspace/heating-up-dbs/explorations/mnist/saves/'
    for p in Path(path).rglob('*.npy'):
        print(p)
        arr = np.load(str(p))
        print(arr.shape)


def extract_plot_ranges(X, pad=0.5):
    """Extract plot ranges given 1D arrays of X and Y axis values."""
    min_x1, max_x1 = np.min(X[:, 0]) - pad, np.max(X[:, 0]) + pad
    min_x2, max_x2 = np.min(X[:, 1]) - pad, np.max(X[:, 1]) + pad
    return min_x1, max_x1, min_x2, max_x2


def generate_grid_points(min_x, max_x, min_y, max_y, resolution=100):
    """Generate resolution * resolution points within a given range."""
    xx, yy = np.meshgrid(np.linspace(min_x, max_x, resolution),
                         np.linspace(min_y, max_y, resolution))
    return np.c_[xx.ravel(), yy.ravel()]


def demo_simple_decision_boundary():
    # https://github.com/vrjkmr/decision-boundary/
    # Load data
    data = load_iris()

    # Obtain data, target
    X, y = data.data, data.target
    X_train = X[y != 2]
    y_train = y[y != 2]
    X_test = X[y == 2]
    y_test = y[y == 2]

    fs = FeaturesStorage('iris')

    fs.add('trainval', {
        'feats': X_train,
        'target': y_train
    })

    fs.add('test', {
        'feats': X_test,
        'target': y_test
    })
    training_feats = fs.training_feats
    o_data = OpensetData(features=training_feats['feats'], class_labels=training_feats['target'])

    approach = 'EVM'
    algorithm_parameters = "--distance_metric euclidean --tailsize 0.5"
    saver_parameters = f"--OOD_Algo {approach}"
    openset_model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

    model = OpensetTrainer(o_data, openset_model_params)
    model.train()

    y_train_incl = model.predict(X_train)
    y_train_incl_class = model.predict_classes(X_train)

    y_test_incl = model.predict(X_test)
    y_test_incl_class = model.predict_classes(X_test)

    print(f'Correct included: {torch.count_nonzero(y_train_incl)} / 100')
    print(f'False included: {torch.count_nonzero(y_test_incl)} / 50')

    # 3. transform dataset into the 2D space
    # mapper = TSNE(n_components=2, random_state=13)
    mapper = umap.UMAP(verbose=True, densmap=False, n_epochs=60)
    X_train_2d = mapper.fit_transform(X_train)
    X_test_2d = mapper.transform(X_test)

    voronoi = KNeighborsClassifier(n_neighbors=1).fit(X_train_2d, y_train_incl)

    # build grid
    min_x1, max_x1, min_x2, max_x2 = extract_plot_ranges(X_train_2d)
    grid_points = generate_grid_points(min_x1, max_x1, min_x2, max_x2)
    print("Grid points: {}".format(grid_points.shape))

    # get grid predictions
    background_predictions = voronoi.predict(grid_points)
    print("Background predictions: {}".format(background_predictions.shape))

    # 6. plot decision boundaries
    fig, ax = plt.subplots(1, 1, figsize=(5, 5.5))

    # plot grid and dataset
    # set colormap
    CMAP = "tab10"

    # ax.set_title("20-NN (test_acc: {:.2f})".format(test_accuracy))
    ax.scatter(grid_points[:, 0], grid_points[:, 1], c=background_predictions,
               cmap=CMAP, alpha=0.4, s=4)
    scatter1 = ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train_incl_class, s=40, cmap=CMAP)

    scatter2 = ax.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test_incl_class, s=40, cmap=CMAP, marker='^')

    # set axis parameters
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_xlim([min_x1, max_x1])
    ax.set_ylim([min_x2, max_x2])
    plt.tight_layout(pad=2.)
    plt.show()


if __name__ == "__main__":
    demo_simple_decision_boundary()
