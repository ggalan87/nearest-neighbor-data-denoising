import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.graph_objects as go

import torch
from losses import TripletLoss

"""
The script generates some samples (2D) from manually specified distributions (normal) for some classes, in order to 
simulate the situation that features of samples are mostly separable however mixed in some cases. This is in order to
get better insight about what is happening in real situations with the loss, more specifically triplet loss and
variances with respect e.g. to the margin.
"""


def plot_data_points(data_points, class_names, interactive, x_range, y_range, texts=None, filename='figure.html'):
    if interactive:
        fig = go.Figure()

        for i, d in enumerate(data_points):
            if not texts:
                text = [f'{class_names[i]}{j}' for j in range(d.shape[0])]
            else:
                text = texts[i]

            fig.add_trace(go.Scatter(x=d[:, 0], y=d[:, 1],
                                     text=text,
                                     mode='markers+text', marker=dict(size=20, color=i),
                                     name=class_names[i]))
            fig.update_xaxes(range=x_range)
            fig.update_yaxes(range=y_range)

        fig.write_html(filename)

    else:
        for i, d in enumerate(data_points):
            plt.plot(d[:, 0], d[:, 1], 'o', color=list(mcolors.BASE_COLORS.values())[i])

        # for i in range(X1.shape[0]):
        #     plt.annotate(f'A{i}', (X1[i, 0], X1[i, 1]), color='magenta')
        #     plt.annotate(f'B{i}', (X2[i, 0], X2[i, 1]), color='magenta')

        plt.show()


def get_data(force_new):
    data_points = []
    if not os.path.exists(data_path) or force_new:
        means = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) + 1.0  # positive only
        cov = np.array([[0.2, 0.0], [0.0, 0.2]])
        for mu in means:
            data_points.append(np.random.multivariate_normal(mu, cov, 50))

        with open(data_path, 'wb') as f:
            for d in data_points:
                np.save(f, d)
    else:
        with open(data_path, 'rb') as f:
            for _ in range(n_classes):
                data_points.append(np.load(f))
    return data_points


def filter_data_points(data_points, selected_indices):
    filtered_data_points = []

    for i, d in enumerate(data_points):
        filtered_data_points.append(d[selected_indices[i]])
    return filtered_data_points


def indices_to_texts(class_names, selected_indices):
    texts = []
    for i, c in enumerate(class_names):
        texts.append([f'{c}{si}' for si in selected_indices[i]])
    return texts


rng = np.random.RandomState(0)

interactive = True
force_new = False
filter_points = False

selected_indices_bad = [[], [26, 0, 5, 11], [27, 0, 5, 8], []]
selected_indices_good = [[], [34, 27, 7, 29], [19, 44, 23, 39], []]

selected_indices = selected_indices_bad

data_path = './data.npy'
class_names = ['A', 'B', 'C', 'D']
n_classes = len(class_names)
data_points = get_data(force_new)

if filter_points:
    data_points = filter_data_points(data_points, selected_indices)

x_range = [0, 4]
y_range = [0, 4]

plot_data_points(data_points, class_names, interactive, x_range, y_range,
                 indices_to_texts(class_names, selected_indices) if filter_points else None, 'figure.html')

with torch.no_grad():
    loss = TripletLoss(margin=0.3)

    data_points = [torch.Tensor(d) for d in data_points if len(d) > 0]
    targets = []
    for i, d in enumerate(data_points):
        targets.extend([i] * d.shape[0])
    targets = torch.Tensor(targets)
    inputs = torch.vstack(data_points)

    loss = loss.forward(inputs, targets)
    print(loss)
