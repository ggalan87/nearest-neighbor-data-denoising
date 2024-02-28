# Visualizations
This is a package that is related to visualizations of various forms.

## Embeddings visualization
Visualization of embeddings is done using UMAP library in order to compute features projections from N-D to 2-D such that they can be plotted using scatter plots. This functionality is implemented in [embeddings_visualization.py](embeddings_visualization.py). More specifically an `EmbeddingsVisualizer` instance accepts features specified in FeaturesStorage format (see [features_storage.py](../features_storage.py)). It either accepts a single instance or multiple instances given in a dictionary. In the first case it computes UMAP. In the later case it computes AlignedUMAP using all features simultaneously. Currently only matplotlib backend is implemented. Currently UMAP and plotting parameters are default / hardcoded / not exposed. Some util functions are defined in [umap_utils.py](umap_utils.py).

## Model Interpreter
Interpretability of trained models stands for visualization of the impact of various parts of the model to the predicted class. An intuitive interpretability measure for example is to visualize the impact of input pixels to the prediction of a specific class. Otherwise, which parts of the image contributed to the prediction. `ModelInterpreter` class defined in [model_interpreter.py](model_interpreter.py) is a fronted to [capture.ai](https://capture.ai) model interpretability library. It also visualizes the results using matplotlib.

## Model  Visualizer
Visualization of models stands for visualizing the architecture, otherwise a graph which shows the interconnections between the layers of the model. `ModelVisualizer` defined in [model_visualizer.py](model_visualizer.py) is a small frontend to the [torchviz](https://github.com/szagoruyko/pytorchviz) library.

## Rest files
| filename                                                                           | description                                                                                                   |
|:-----------------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------|
| [plot_umap.py](samples/plot_umap.py), [plotly_sample.py](samples/plotly_sample.py) | Some testing scripts                                                                                          |
| [space_viz.py](space_viz.py)                                                       | Visualization of a sample space in order to get a better insight about the triplet loss, and effect of margin |
| [test_aug.py](samples/test_aug.py)                                                 | Empty file to test augly library                                                                              |
| rest in `samples` dir                                                              | Samples which demonstrate various usages from various libs                                                    |

