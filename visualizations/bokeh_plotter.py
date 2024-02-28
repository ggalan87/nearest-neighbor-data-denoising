import pandas as pd
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, ImageURL, CDSView, IndexFilter
from bokeh.transform import factor_cmap, factor_mark
from bokeh.palettes import *


class BokehPlotter:
    tooltip_template = \
        """
        <div>
            <div>
                <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
            </div>
            <div>
                <span style='font-size: 16px; color: #224499'>Class:</span>
                <span style='font-size: 18px'>@class_name</span>
                <span style='font-size: 18px'>(@label)</span>
            </div>
        </div>
        """

    def __init__(self):
        pass

    def plot(self,
             plot_title: str,
             embeddings_2d,
             labels,
             names_dict: Dict[int, str],
             image_server_paths,
             output_filepath,
             predicted_labels=None,
             noisy_samples=None):
        """
        TODO: Refactor to take the dataframe directly

        :param plot_title:
        :param embeddings_2d:
        :param labels:
        :param names_dict:
        :param image_server_paths:
        :param output_filepath:
        :param predicted_labels:
        :param noisy_samples:
        :return:
        """
        output_file(output_filepath)

        unique_classes = np.unique(labels)

        dataframe = pd.DataFrame(embeddings_2d, columns=('x', 'y'))

        dataframe['label'] = [str(x) for x in labels]
        names = [names_dict[label] for label in labels]

        # TODO: Decide how to plot gt vs pred, e.g. (a) in same plot, but which color, (b) two plots,
        #  (c) JS callback and change color of points
        if predicted_labels is not None:
            raise NotImplementedError('Functionality for plotting predicted labels along with '
                                      'ground truth is not implemented yet')

        if noisy_samples is not None:
            dataframe['is_noisy'] = list(map(str, noisy_samples))
            is_noisy = ['False', 'True']
            markers = ['circle', 'triangle']
            marker = factor_mark('is_noisy', markers, is_noisy)
        else:
            marker = 'circle'

        dataframe['class_name'] = names

        # TODO: Implement appropriate palette mapping based on number of classes, e.g. mapping to more
        palette_class = Spectral10 if len(unique_classes) <= 10 else inferno(len(unique_classes))  # Inferno256
        # color_mapping = CategoricalColorMapper(factors=[str(len(unique_classes) - x) for x in unique_classes],
        #                                        palette=palette_class)

        color_mapping = CategoricalColorMapper(factors=[str(x) for x, _ in enumerate(unique_classes)],
                                               palette=palette_class)

        plot_figure = figure(
            title=plot_title,
            width=1500,
            height=1000,
            tools=('pan, wheel_zoom, reset, save')
        )

        if image_server_paths:
            dataframe['image'] = image_server_paths

            plot_figure.add_tools(HoverTool(tooltips=self.tooltip_template))

        datasource = ColumnDataSource(dataframe)

        color = dict(field='label', transform=color_mapping)

        for i, class_id in enumerate(unique_classes):
            # TODO: Check why in the updated bokeh version numpy array needs to be converted to list, else no plot is
            #  shown. Maybe it has to do with int64 vs int representation and some overflow resulting to index error ?
            view = CDSView(filter=IndexFilter(np.where(labels == class_id)[0].tolist()))

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
                legend_label=names_dict[class_id],
                marker=marker
            )

        plot_figure.legend.location = "top_right"
        plot_figure.legend.click_policy = "mute"
        plot_figure.legend.nrows = 20
        plot_figure.legend.ncols = 5

        plot_figure.add_layout(plot_figure.legend[0], 'right')

        show(plot_figure)
