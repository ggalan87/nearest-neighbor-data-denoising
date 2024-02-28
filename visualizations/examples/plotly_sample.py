import pandas as pd
import plotly.graph_objs as go

"""file:///run/user/1000/gvfs/sftp:host=amihome2-ubuntu,user=amidemo/home/amidemo/devel/workspace/object_classifier_deploy/visualizations/images/20190921-00001-01-01.jpg"""

df = pd.DataFrame({"x":[0,1],
                  "y":[0,1],
                   "text":["<a href=\"file:///run/user/1000/gvfs/sftp:host=amihome2-ubuntu,user=amidemo/home/amidemo/devel/workspace/object_classifier_deploy/visualizations/images/20190921-00001-01-01.jpg\">name1</a>",
                           "<a href=\"https://google.com\">name2</a>"]})
fig = go.Figure()
fig.add_trace(
    go.Scatter(x=df["x"],
               y=df["y"],
               mode="markers+text",
               # Just pick one of the two
               hovertext=df["text"],
               text=df["text"],
               textposition="top center",
               textfont_size=8))

fig.write_html(f'hyperlink_fig.html')