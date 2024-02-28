import numpy as np
import umap
from sklearn.datasets import load_digits
from io import BytesIO
from PIL import Image
import base64
import pandas as pd
from bokeh.plotting import figure, show, output_notebook, output_file
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, ImageURL, CDSView, IndexFilter
from bokeh.palettes import Spectral10, Viridis256, linear_palette
from bokeh.transform import factor_cmap, factor_mark
from bokeh.models import FreehandDrawTool

output_file("/media/amidemo/Data/object_classifier_data/global_plots/sample_plots/index.html")


def embeddable_image(data):
    img_data = 255 - 15 * data.astype(np.uint8)
    image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
    buffer = BytesIO()
    image.save(buffer, format='png')
    for_encoding = buffer.getvalue()
    return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()


digits = load_digits()
# print(digits.DESCR)

reducer = umap.UMAP(random_state=42)
reducer.fit(digits.data)

embedding = reducer.transform(digits.data)

# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))


digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
digits_df['digit'] = [str(x) for x in digits.target]
digits_df['odd'] = ['True' if x % 2 == 0 else 'False' for x in digits.target]
digits_df['image'] = list(map(embeddable_image, digits.images))

datasource = ColumnDataSource(digits_df)
color_mapping = CategoricalColorMapper(factors=[str(9 - x) for x in digits.target_names],
                                       palette=linear_palette(Viridis256, 10))


plot_figure = figure(
    title='UMAP projection of the Digits dataset',
    width=1000,
    height=1000,
    tools=('pan, wheel_zoom, reset')
)

renderer = plot_figure.multi_line([[1, 9]], [[5, 5]], line_width=5, alpha=0.4, color='red')
draw_tool = FreehandDrawTool(renderers=[renderer], num_objects=3)
plot_figure.add_tools(draw_tool)
plot_figure.toolbar.active_drag = draw_tool


plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Digit:</span>
        <span style='font-size: 18px'>@digit</span>
    </div>
</div>
"""))

color = dict(field='digit', transform=color_mapping)

is_odd = ['True', 'False']
markers = ['circle', 'triangle']

unique_targets = np.unique(digits.target)
for i, ut in enumerate(unique_targets):
    view = CDSView(filter=IndexFilter(np.where(digits.target == ut)[0].tolist()))

    plot_figure.scatter(
        'x',
        'y',
        source=datasource,
        color=color,
        line_alpha=0.6,
        fill_alpha=0.6,
        size=12,
        muted_color=color,
        muted_alpha=0.05,
        view=view,
        legend_label=str(i),
        marker=factor_mark('odd', markers, is_odd)
    )

plot_figure.legend.location = "top_left"
plot_figure.legend.click_policy = "mute"
plot_figure.legend.nrows = 5
plot_figure.legend.ncols = 2

plot_figure.add_layout(plot_figure.legend[0], 'right')

show(plot_figure)

