import itertools
from typing import List, Dict

import hvplot
import numpy as np
import torch
import pandas as pd

from evt.vast_openset import OpensetData, OpensetTrainer, OpensetModelParameters
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, ImageURL, CDSView, IndexFilter
from bokeh.transform import factor_cmap, factor_mark, linear_cmap
from bokeh.palettes import Spectral10, inferno, Category10_10

from holoviews.element.tiles import EsriImagery
import hvplot.pandas
import hvplot
import holoviews as hv, colorcet as cc
import panel as pn

from lightning.data.dataset_utils import disturb_targets_symmetric


class OpensetModelPlotter:
    def __init__(self, trainer: OpensetTrainer):
        self._trainer = trainer

    def _prepare_data_frame(self, data: OpensetData):
        dict_data = data.features_dict

        data = torch.cat(list(dict_data.values()), dim=0)
        assert data.shape[1] == 2

        labels = []
        colors = []
        for k in dict_data:
            label = int(k)  # remove trailing zeros
            labels.extend([label] * dict_data[k].shape[0])
            colors.extend([Category10_10[label]] * dict_data[k].shape[0])

        dataframe = pd.DataFrame(data, columns=('x', 'y'))
        dataframe['label'] = labels
        dataframe['label_color'] = colors
        return dataframe

    def _compute_global_heatmap(self, pos_features_x, pos_features_y, plot_size=200, *args, **kwargs):
        min_x, max_x = np.min(pos_features_x), np.max(pos_features_x)
        min_y, max_y = np.min(pos_features_y), np.max(pos_features_y)
        x = np.linspace(min_x * 1.5, max_x * 1.5, plot_size)
        y = np.linspace(min_y * 1.5, max_y * 1.5, plot_size)
        pnts = list(itertools.chain(itertools.product(x, y)))
        pnts = np.array(pnts)

        res = self._trainer._get_probs(pnts, *args, **kwargs)
        heatmap = np.array(res).reshape(plot_size, plot_size).transpose()

        xx, yy = np.meshgrid(x, y)

        df = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'z': heatmap.ravel()})

        return df

    def _compute_labels_heatmaps(self, pos_features_x, pos_features_y, plot_size=200, *args, **kwargs):
        min_x, max_x = np.min(pos_features_x), np.max(pos_features_x)
        min_y, max_y = np.min(pos_features_y), np.max(pos_features_y)
        x = np.linspace(min_x * 1.5, max_x * 1.5, plot_size)
        y = np.linspace(min_y * 1.5, max_y * 1.5, plot_size)
        pnts = list(itertools.chain(itertools.product(x, y)))
        pnts = np.array(pnts)

        training_labels = self._trainer.data.labels_to_indices.keys()

        labels_heatmaps = {}
        for l in training_labels:

            l_idx = self._trainer.data.labels_to_indices[l]
            res = self._trainer._get_probs(pnts, single_probability=False)
            heatmap = np.array(res[:, l_idx]).reshape(plot_size, plot_size).transpose()

            xx, yy = np.meshgrid(x, y)
            labels_heatmaps[l] = pd.DataFrame({'x': xx.ravel(), 'y': yy.ravel(), 'z': heatmap.ravel()})

        return labels_heatmaps

    def plot_heatmap(self, data: OpensetData):
        def update_plot(df, label_heatmap_plots: Dict, selected_labels):
            selected_indices = pd.Series(np.zeros(shape=(len(df),), dtype=bool))

            unique_labels = np.unique(df['label'].to_numpy())

            for l in unique_labels:
                if l in selected_labels:
                    selected_indices |= (df['label'] == l)

            xlim = df['x'].min(), df['x'].max()
            ylim = df['y'].min(), df['y'].max()
            filtered_df = df[selected_indices]

            data_plot = filtered_df.hvplot.scatter(x='x', y='y', size=10, alpha=0.7, xlabel="f1", ylabel="f2",
                                                   xlim=xlim, ylim=ylim, color='label_color')\
                .opts(width=plot_size, height=plot_size)

            if len(selected_labels) == 0:
                return data_plot
            else:
                heatmap_plot = label_heatmap_plots[selected_labels[0]]
                for l in selected_labels[1:]:
                    heatmap_plot *= label_heatmap_plots[l]
                return heatmap_plot * data_plot


        data_df = self._prepare_data_frame(data)
        test_labels = data_df['label'].to_numpy()
        unique_test_labels = np.unique(test_labels)

        n_model_classes = len(self._trainer.data.features_dict)
        n_test_classes = len(data.features_dict)

        palette_class = Spectral10 if n_test_classes <= 10 else inferno(n_test_classes)  # Inferno256
        color_mapping = CategoricalColorMapper(factors=[str(x) for x, _ in enumerate(unique_test_labels)],
                                               palette=palette_class)

        plot_size = 400

        heatmap_df = self._compute_global_heatmap(data_df['x'].to_numpy(), data_df['y'].to_numpy(), plot_size=plot_size)
        labels_heatmaps_df = self._compute_labels_heatmaps(data_df['x'].to_numpy(), data_df['y'].to_numpy(), plot_size=plot_size)

        # data_plot = data_df.hvplot.scatter(x='x', y='y', size=10, alpha=0.7, xlabel="f1", ylabel="f2",
        #                                    c='label', cmap='Category10')\
        #     .opts(width=plot_size, height=plot_size)

        global_heatmap_plot = labels_heatmaps_df[0].hvplot.heatmap(
            'x',
            'y',
            'z',
            rasterize=True,
            cmap=cc.rainbow4,
            colorbar=True).opts(width=plot_size, height=plot_size, colorbar_position='bottom')

        plots = {}
        for label, l_heatmap_df in labels_heatmaps_df.items():
            plot = l_heatmap_df.hvplot.heatmap(
                'x',
                'y',
                'z',
                rasterize=True,
                cmap=cc.rainbow4,
                colorbar=True).opts(width=plot_size, height=plot_size, colorbar_position='bottom')
            plots[label] = plot

        multi_select = pn.widgets.MultiSelect(name='MultiSelect', value=list(data_df['label'].unique()),
                                              options=list(data_df['label'].unique()), size=8)

        # Create a reactive plot that updates based on the selected label
        reactive_plot = pn.bind(update_plot, df=data_df, label_heatmap_plots=plots,
                                selected_labels=multi_select)

        # Combine the widget and reactive plot into a Panel layout
        layout = pn.Column(multi_select, reactive_plot)

        layout.show(port=43173)

        # hvplot.show(plot * data_plot, port=43173)


def sample_data_plot():
    training_data = pd.read_csv('../../../evt/TestData/train_mnist.csv').to_numpy()
    training_features = torch.Tensor(training_data[:, 1:3])
    training_labels = torch.IntTensor(training_data[:, 0])
    random_indices = torch.randperm(len(training_labels))[:int(0.3 * len(training_labels))]
    training_features = training_features[random_indices]
    training_labels = training_labels[random_indices]
    noisy_training_labels, noisy_indices = disturb_targets_symmetric(training_labels, perc=0.5)


    testing_data = pd.read_csv('../../../evt/TestData/test_mnist.csv').to_numpy()
    testing_features = torch.Tensor(testing_data[:, 1:3])
    testing_labels = torch.IntTensor(testing_data[:, 0])

    training_data = OpensetData(features=training_features, class_labels=training_labels)
    noisy_training_data = OpensetData(features=training_features, class_labels=noisy_training_labels)
    testing_data = OpensetData(features=testing_features, class_labels=testing_labels)

    # approach, algorithm_parameters = \
    #     ('OpenMax', "--distance_metric euclidean --distance_multiplier 0.7 --tailsize 0.55")
    approach, algorithm_parameters = \
        ('EVM', "--distance_metric euclidean --distance_multiplier 0.7 --tailsize 0.10 --cover_threshold 0.7")
    saver_parameters = f"--OOD_Algo {approach}"
    model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

    trainer = OpensetTrainer(noisy_training_data, model_params)
    trainer.train()
    # trainer.eval_classification(testing_data)

    omp = OpensetModelPlotter(trainer)
    omp.plot_heatmap(noisy_training_data)


if __name__ == '__main__':
    sample_data_plot()
