import numpy as np
import sklearn.datasets
import umap
import umap.plot
import umap.utils as utils
import umap.aligned_umap
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import plotly.graph_objects as go

from umap_utils import *


def axis_bounds(embedding):
    """
    Computes bounds for axes to use them while plotting UMAP
    :param embedding: The embeddings from which to compute the bounds
    :return: the axes bounds
    """
    left, right = embedding.T[0].min(), embedding.T[0].max()
    bottom, top = embedding.T[1].min(), embedding.T[1].max()
    adj_h, adj_v = (right - left) * 0.1, (top - bottom) * 0.1
    return [left - adj_h, right + adj_h, bottom - adj_v, top + adj_v]


def plot_umap(feat_c, classes, scenes, views, model_names, inst_ids, inst_pos, img_prefix, interactive=True):

    embeddings_path = Path('umap.npy')

    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
    else:
        # supervised dimensionality reduction using the true classes
        embeddings = compute_embeddings(feat_c, classes)

    if interactive:
        unique_classes = np.unique(classes)

        for j, t in enumerate(model_names):
            fig = go.Figure()

            for i, c in tqdm(enumerate(unique_classes)):
                class_data = embeddings[j][classes == c]
                scenes_data = scenes[classes == c]
                views_data = views[classes == c]
                inst_ids_data = inst_ids[classes == c]
                #inst_pos_data = inst_pos[classes == c]

                # hover info
                hover_info_data = np.vstack((scenes_data, views_data)).T

                # Construct texts list
                texts = []
                for (scene, view), inst_id in zip(hover_info_data, inst_ids_data):
                    texts.append("""<a href="{}{}-{:02d}-{}.jpg">{}</a>""".format(img_prefix, scene, int(view), inst_id, c))
                    # texts.append("""<a href="https://plot.ly/">{}</a>""".format(c))

                fig.add_trace(go.Scatter(x=class_data[:, 0], y=class_data[:, 1],
                                         text=texts,
                                         customdata=hover_info_data,
                                         hovertemplate='%{customdata[0]}-%{customdata[1]:02d}.jpg',
                                         mode='markers+text', marker=dict(size=20, color=i),
                                         name=str(c)))

            fig.write_html(f'/media/amidemo/Data/object_image_viz/MessyTable/umap_embedding_{t}.html')
    else:
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        ax_bound = axis_bounds(np.vstack(embeddings))
        for i, ax in tqdm(enumerate(axs.flatten())):
            scatter = ax.scatter(*embeddings[i].T, s=2, c=classes, cmap="Spectral", alpha=0.3)
            ax.axis(ax_bound)
            ax.set(xticks=[], yticks=[])


        plt.tight_layout()
        plt.show()


def load_messy_table_feats(subset=None):
    # Load test_easy data from MessyTable dataset, 2 model versions, multi gpu training (original) and single gpu training
    raw_feats_single_path = Path('/home/amidemo/devel/workspace/MessyTable/models/asnet_1gpu/raw_feats.npz')
    raw_feats_multi_path = Path('/home/amidemo/devel/workspace/MessyTable/models/asnet/raw_feats.npz')

    assert raw_feats_single_path.exists() and raw_feats_multi_path.exists()

    # Load corresponding information
    raw_feats_single = np.load(raw_feats_single_path)
    raw_feats_multi = np.load(raw_feats_multi_path)

    # Get the actual features
    feat_c_single = raw_feats_single['feat_c']
    feat_c_multi = raw_feats_multi['feat_c']

    # Get the rest information, which is common for all models
    classes = raw_feats_multi['classes']
    scenes = raw_feats_multi['scenes']
    views = raw_feats_multi['views']
    inst_ids = raw_feats_multi['inst_ids']
    inst_pos = raw_feats_multi['inst_pos']

    # Compute the subsets / needed to investigate smaller portion of the data. The subset is by using fewer classes
    indices = np.zeros_like(classes, dtype=np.bool8)  # initially we select none of the classes

    if subset is None:
        subset = range(1, len(np.unique(classes)) + 1)

    # afterwards we select the indices which correspond to the classes contained in the desired subset
    classes_subset_range = subset
    for c in classes_subset_range:
        indices |= classes == c

    print(f'Initial set: {classes.shape[0]} samples.')

    classes = classes[indices]
    feat_c_single = feat_c_single[indices]
    feat_c_multi = feat_c_multi[indices]
    scenes = scenes[indices]
    views = views[indices]
    inst_ids = inst_ids[indices]
    inst_pos = inst_pos[indices]

    print(f'Reduced set: {classes.shape[0]} samples.')

    feat_c = [feat_c_single, feat_c_multi]
    model_names = ['single', 'multi']
    img_prefix =\
        'http://139.91.96.135:8345/MessyTable/tmp_crops/'
    return feat_c, classes, scenes, views, model_names, inst_ids, inst_pos, img_prefix


def main():
    interactive = True  # True for interactive plotting using plotly, otherwise matplotlib (less informative)
    subset = range(1, 31)  # classes 1-30
    feat_c, classes, scenes, views, model_names, inst_ids, inst_pos, img_prefix = load_messy_table_feats(subset)

    plot_umap(feat_c, classes, scenes, views, model_names, inst_ids, inst_pos, img_prefix, interactive)


if __name__ == '__main__':
    main()

