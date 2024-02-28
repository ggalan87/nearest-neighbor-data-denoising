from pathlib import Path
import pickle

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

from visualizations.embeddings_visualization import EmbeddingsVisualizer

aligned_embeddings_path = Path('../../lightning/embeddings/market1501/orig_triloss/umap_aligned_embeddings.pkl')
aligned_embeddings = pickle.load(open(aligned_embeddings_path, 'rb'))

per_epoch_embeddings = np.stack([emb['train']['data'] for emb in aligned_embeddings])
labels = aligned_embeddings[0]['train']['labels']

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

anim.save("aligned_umap_orig_triloss.gif", writer="pillow")
plt.close(anim._fig)
