import platform
from bokeh.plotting import ColumnDataSource, figure, output_file, show

output_file("/home/amidemo/devel/workspace/object_classifier_deploy/lightning/pipelines_datasets_scripts/mnist/lightning_logs/dirtymnist_ResNetMNIST/version_7/embeddings/toolbar.html")

images_server = f'http://{platform.node()}:8345'

source = ColumnDataSource(data=dict(
    x=[1, 2, 3, 4, 5],
    y=[2, 5, 8, 2, 7],
    desc=['A', 'b', 'C', 'd', 'E'],
    imgs=[
        f'{images_server}/dirtymnist/images/test_1.png',
        f'{images_server}/dirtymnist/images/test_2.png',
        f'{images_server}/dirtymnist/images/test_3.png',
        f'{images_server}/dirtymnist/images/test_4.png',
        f'{images_server}/dirtymnist/images/test_5.png',
    ],
    fonts=[
        '<i>italics</i>',
        '<pre>pre</pre>',
        '<b>bold</b>',
        '<small>small</small>',
        '<del>del</del>'
    ]
))

TOOLTIPS = """
    <div>
        <div>
            <img
                src="@imgs" height="42" alt="@imgs" width="42"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@desc</span>
            <span style="font-size: 15px; color: #966;">[$index]</span>
        </div>
        <div>
            <span>@fonts{safe}</span>
        </div>
        <div>
            <span style="font-size: 15px;">Location</span>
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
    </div>
"""

p = figure(width=400, height=400, tooltips=TOOLTIPS,
           title="Mouse over the dots")

p.circle('x', 'y', size=20, source=source)

show(p)