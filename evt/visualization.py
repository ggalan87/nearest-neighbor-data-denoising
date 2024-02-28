import itertools

import numpy as np
import torch
import pandas as pd

from evt.vast_openset import OpensetData, OpensetTrainer, OpensetModelParameters
import pandas as pd
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, ImageURL, CDSView, IndexFilter
from bokeh.transform import factor_cmap, factor_mark, linear_cmap
from bokeh.palettes import *


class OpensetModelPlotter:
    def __init__(self, trainer: OpensetTrainer):
        self._trainer = trainer

    def _prepare_data_frame(self, data: OpensetData):
        dict_data = data.features_dict

        data = torch.cat(list(dict_data.values()), dim=0)
        assert data.shape[1] == 2

        labels = []
        for k in dict_data:
            label = str(int(k))  # remove trailing zeros
            labels.extend([label] * dict_data[k].shape[0])

        dataframe = pd.DataFrame(data, columns=('x', 'y'))
        dataframe['label'] = labels
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
        # Create a ColumnDataSource with the data
        source = ColumnDataSource(data=dict(x=xx.ravel(), y=yy.ravel(), z=heatmap.ravel()))

        return source

    def plot_heatmap(self, data: OpensetData):
        output_file('./openset_plots/histogram.html')

        data_frame = self._prepare_data_frame(data)
        test_labels = data_frame['label'].to_numpy()
        unique_test_labels = np.unique(test_labels)

        n_model_classes = len(self._trainer.data.features_dict)
        n_test_classes = len(data.features_dict)

        palette_class = Spectral10 if n_test_classes <= 10 else inferno(n_test_classes)  # Inferno256
        color_mapping = CategoricalColorMapper(factors=[str(x) for x, _ in enumerate(unique_test_labels)],
                                               palette=palette_class)

        plot_size = 1000

        heatmap = self._compute_global_heatmap(data_frame['x'].to_numpy(), data_frame['y'].to_numpy(),
                                               plot_size=plot_size)


        plot_figure = figure(
            title='heatmap',
            plot_width=plot_size,
            plot_height=plot_size,
            tools=('pan, wheel_zoom, reset, save')
        )

        # Generate some sample data
        # x = np.arange(0, 10, 0.1)
        # y = np.arange(0, 10, 0.1)
        # xx, yy = np.meshgrid(x, y)
        # z = np.sin(xx) * np.cos(yy)

        # Create a ColumnDataSource with the data
        #source = ColumnDataSource(data=dict(x=xx.ravel(), y=yy.ravel(), z=z.ravel()))

        # Create a color mapper for the patches
        mapper = linear_cmap(field_name='z', palette='Inferno256', low=0.0, high=1.0)

        # Add the patches to the figure
        plot_figure.rect('x', 'y', width=0.1, height=0.1, source=heatmap, fill_color=mapper,
                         line_width=0, line_color=None)
        plot_figure.grid.visible = False
        show(plot_figure)
        return
        datasource = ColumnDataSource(data_frame)

        color = dict(field='label', transform=color_mapping)

        for i, class_id in enumerate(unique_test_labels):
            view = CDSView(source=datasource, filters=[IndexFilter(np.where(test_labels == class_id)[0])])

            plot_figure.scatter(
                'x',
                'y',
                source=datasource,
                color=palette_class[i],
                line_alpha=0.6,
                fill_alpha=0.6,
                size=12,
                muted_color=color,
                muted_alpha=0.05,
                view=view,
                legend_label=str(class_id),
                marker='circle'
            )

        plot_figure.legend.location = "top_right"
        plot_figure.legend.click_policy = "mute"
        show(plot_figure)


def sample_data_plot():
    training_data = pd.read_csv('./TestData/train_mnist.csv').to_numpy()
    training_features = torch.Tensor(training_data[:, 1:3])
    training_labels = torch.IntTensor(training_data[:, 0])

    testing_data = pd.read_csv('./TestData/test_mnist.csv').to_numpy()
    testing_features = torch.Tensor(testing_data[:, 1:3])
    testing_labels = torch.IntTensor(testing_data[:, 0])

    training_data = OpensetData(features=training_features, class_labels=training_labels)
    testing_data = OpensetData(features=testing_features, class_labels=testing_labels)

    approach, algorithm_parameters = \
        ('OpenMax', "--distance_metric euclidean --distance_multiplier 0.7 --tailsize 0.55")
    saver_parameters = f"--OOD_Algo {approach}"
    model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

    trainer = OpensetTrainer(training_data, model_params)
    trainer.train()
    # trainer.eval_classification(testing_data)
    #trainer.plot(testing_data)

    omp = OpensetModelPlotter(trainer)
    omp.plot_heatmap(testing_data)


if __name__ == '__main__':
    # https://towardsdatascience.com/plotting-heat-maps-in-python-using-bokeh-folium-and-hvplot-eb7c7f49dbc6
    sample_data_plot()
