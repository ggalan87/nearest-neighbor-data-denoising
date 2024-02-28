from bokeh.plotting import figure, show, output_file
from bokeh.sampledata.iris import flowers as df
from bokeh.transform import factor_cmap, factor_mark

output_file("/media/amidemo/Data/object_classifier_data/global_plots/sample_plots/index.html")

SPECIES = ['setosa', 'versicolor', 'virginica']
MARKERS = ['hex', 'circle_x', 'triangle']
CIDS = ['0', '1', '2']

p = figure(tooltips="species: @species")
color = factor_cmap('species', 'Category10_3', SPECIES)

df['cid'] = ['0'] * 50 + ['1'] * 50 + ['2'] * 50

fm = factor_mark('cid', MARKERS, CIDS)

p.scatter("petal_length", "sepal_width", source=df, legend="species", alpha=0.5,
          size=12, color=color, muted_color=color, muted_alpha=0.2, marker=factor_mark('cid', MARKERS, CIDS),)

p.legend.location = "top_left"
p.legend.click_policy = "mute"

show(p)
