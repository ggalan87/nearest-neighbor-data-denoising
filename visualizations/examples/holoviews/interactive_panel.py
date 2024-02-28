import numpy as np
import pandas as pd
import hvplot.pandas
import panel as pn

# Create a DataFrame
df = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                   'y': [2, 4, 6, 8, 10],
                   'label': ['A', 'B', 'A', 'B', 'A']})


# Define a function for updating the filtered plot
def update_plot(selected_labels):
    selected_indices = pd.Series(np.zeros(shape=(len(df), ), dtype=bool))

    unique_labels = np.unique(df['label'].to_numpy())

    for l in unique_labels:
        if l in selected_labels:
            selected_indices |= (df['label'] == l)

    filtered_df = df[selected_indices]
    plot = filtered_df.hvplot.scatter(x='x', y='y')
    return plot
#     selected_labels = label_widget.value
#     df['visible'] = df['label'].isin(selected_labels)
#     plot = df[df['visible']].hvplot.scatter(x='x', y='y', c='label', cmap='Category10')
#     reactive_plot.object = plot


# Create a widget for selecting the label
label_widget = pn.widgets.Select(options=['All'] + list(df['label'].unique()), name='Label')

multi_select = pn.widgets.MultiSelect(name='MultiSelect', value=list(df['label'].unique()),
                                      options=list(df['label'].unique()), size=8)

# Create a reactive plot that updates based on the selected label
reactive_plot = pn.bind(update_plot, selected_labels=multi_select)

# Combine the widget and reactive plot into a Panel layout
layout = pn.Column(multi_select, reactive_plot)

# Display the layout
layout.show(port=43173)
