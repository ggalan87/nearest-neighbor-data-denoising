from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd

from datashader.utils import lnglat_to_meters
from holoviews.element.tiles import EsriImagery
import hvplot.pandas
import hvplot
import holoviews as hv, colorcet as cc

hv.extension('bokeh')

wildfires_dataset_sql_path = Path('/media/amidemo/Data/US_Wildfires/FPA_FOD_20170508.sqlite')
conn = sqlite3.connect(wildfires_dataset_sql_path)

df = pd.read_sql_query("SELECT LATITUDE, LONGITUDE, FIRE_SIZE, STATE, FIRE_YEAR FROM fires", conn)

# Drop states that aren't in continental US
df = df.loc[(df.loc[:, 'STATE'] != 'AK') & (df.loc[:, 'STATE'] != 'HI') & (df.loc[:, 'STATE'] != 'PR')]

dfCopy = df.copy()

dfCopy.loc[:, 'LATITUDE'] = ((dfCopy.loc[:, 'LATITUDE']*10).apply(np.floor))/10
dfCopy.loc[:, 'LONGITUDE'] = ((dfCopy.loc[:, 'LONGITUDE']*10).apply(np.floor))/10
dfCopy.loc[:, 'LatLonRange'] = dfCopy.loc[:, 'LATITUDE'].map(str) + '-' + dfCopy.loc[:, 'LONGITUDE'].map(str)

df_grouped = dfCopy.groupby(['LatLonRange', 'LATITUDE', 'LONGITUDE'])

# Find Number of Fires That Occured in Each Group
# fire_count = df_grouped['FIRE_SIZE'].agg(['count']).reset_index()

# Find Average Fire Size in Each Group
# fire_avgSize = df_grouped['FIRE_SIZE'].agg(['mean']).reset_index()

df.loc[:, 'x'], df.loc[:, 'y'] = lnglat_to_meters(df.LONGITUDE, df.LATITUDE)
# map_tiles = EsriImagery().opts(alpha=0.5, width=700, height=480, bgcolor='black')
plot = df.hvplot(
    'x',
    'y',
    kind='scatter',
    rasterize=True,
    cmap=cc.fire,
    cnorm='eq_hist',
    colorbar=True).opts(colorbar_position='bottom')

hvplot.show(plot, port=43173)

