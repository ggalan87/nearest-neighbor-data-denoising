import os
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import animation
import matplotlib.pyplot as plt
from features_storage import FeaturesStorage
from visualizations.embeddings_visualization import EmbeddingsVisualizer

features_paths = sorted(Path('../../lightning/features/orig').iterdir(), key=os.path.getmtime)

per_epoch_embeddings = []
for i, features_path in enumerate(features_paths):
    # if i % 10 != 0:
    #     continue

    print(f'Processing {str(features_path)}')
    p = Path(features_path)
    if not p.exists():
        raise FileNotFoundError(features_path)

    # Get name from filepath
    model_name = p.stem

    # I need to explicitly specify the target_key because in the context of re-id the target (class label) is the id
    features_storage = FeaturesStorage(cached_path=str(p), target_key='id')

    features_storage.filter_by_ids(keep_range=(1, 15))

    feats, labels = features_storage.raw_features()

    feats2D = PCA(n_components=2).fit_transform(feats[0].numpy())
    labels = labels[0]
    per_epoch_embeddings.append(feats2D)

per_epoch_embeddings = np.stack(per_epoch_embeddings)

ax_bound = EmbeddingsVisualizer.axis_bounds(np.vstack(per_epoch_embeddings))

fig = plt.figure(figsize=(4, 4), dpi=150)
ax = fig.add_subplot(1, 1, 1)

scat = ax.scatter([], [], s=2)
scat.set_array(labels)
scat.set_cmap('Spectral')
text = ax.text(ax_bound[0] + 0.5, ax_bound[2] + 0.5, '')
ax.axis(ax_bound)
ax.set(xticks=[], yticks=[])
plt.tight_layout()

num_frames = per_epoch_embeddings.shape[0]


def animate(i):
    scat.set_offsets(per_epoch_embeddings[i])
    text.set_text(f'Frame {i}')
    return scat


anim = animation.FuncAnimation(
    fig,
    init_func=None,
    func=animate,
    frames=num_frames,
    interval=400)

anim.save("PCA_anim.gif", writer="pillow")
plt.close(anim._fig)
