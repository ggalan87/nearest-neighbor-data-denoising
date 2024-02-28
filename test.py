from pathlib import Path
from PIL import Image
import PIL
import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import torch
import scipy.spatial.distance as scipy_dist
import matplotlib.pyplot as plt
import gc

# root_path = Path('/media/amidemo/Data/cats_n_dogs/Cats_and_Dogs_Breeds_Classification_Oxford_Dataset')
#
# images_path = root_path / 'images/images'
#
# for img_path in images_path.iterdir():
#     if img_path.suffix != '.jpg':
#         continue
#
#     with Image.open(img_path) as im:
#         if not type(im) == PIL.JpegImagePlugin.JpegImageFile:
#             rgb_im = im.convert('RGB')
#             rgb_im.save(img_path, "JPEG")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def jaccard_cont(x1, x2):
    x1_sig = sigmoid(x1)
    x2_sig = sigmoid(x2)

    sim = np.divide(np.sum(np.minimum(x1_sig, x2_sig)),
                     np.sum(np.maximum(x1_sig, x2_sig)))
    return 100 * np.subtract(np.ones_like(sim), sim)


def jaccard_cont_torch(x1, x2):
    x1_sig = torch.sigmoid(x1)
    x2_sig = torch.sigmoid(x2)

    mins = torch.minimum(x1_sig, x2_sig)
    maxs = torch.maximum(x1_sig, x2_sig)

    sim = torch.divide(torch.sum(torch.minimum(x1_sig, x2_sig)),
                        torch.sum(torch.maximum(x1_sig, x2_sig)))
    return torch.subtract(torch.ones_like(sim), sim)


def pairwise_euclidean(inputs):
    n = inputs.size(0)

    # Compute pairwise distance, replace by the official when merged
    dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist


def pairwise_jaccard(inputs):
    n = inputs.size(0)
    dist = torch.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            dist[i, j] = dist[j, i] = jaccard_cont_torch(inputs[i], inputs[j])

    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

    return dist


def compute_pairwise_gallery_query(X_gallery, Y_gallery, X_query, Y_query):
    pids = np.unique(Y_query)

    bins = np.linspace(0, 1, 21)
    intra_class_dists_hist = np.zeros((bins.shape[0] - 1, ))
    inter_class_dists_hist = np.zeros((bins.shape[0] - 1, ))

    dist_mat = scipy_dist.cdist(X_gallery, X_query, metric=jaccard_cont)
    #dist_mat = scipy_dist.cdist(X_gallery, X_query)

    intra_class_dists = []
    inter_class_dists = []
    for i, pid in enumerate(Y_gallery):
       intra_class_dists.append(dist_mat[i][Y_query == pid].ravel())
       inter_class_dists.append(dist_mat[i][Y_query != pid].ravel())

    intra_class_dists = np.concatenate(intra_class_dists)
    inter_class_dists = np.concatenate(inter_class_dists)

    print(np.min(intra_class_dists), np.max(intra_class_dists))
    print(np.min(inter_class_dists), np.max(inter_class_dists))

    plt.hist(intra_class_dists, weights=np.ones(len(intra_class_dists)) / len(intra_class_dists))
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.hist(inter_class_dists, weights=np.ones(len(inter_class_dists)) / len(inter_class_dists), alpha=0.5)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.show()

    pass

def inspect_osnet_feats():
    feat_path = '/media/amidemo/Data/deep_person_reid_logs/orig_log/osnet_x1_0-market1501-softmax/market1501_features.pkl'
    with open(feat_path, 'rb') as f:
        feat = pickle.load(f)

    X_gallery = feat['gallery']['feat']
    X_query = feat['query']['feat']
    Y_gallery = feat['gallery']['pids']
    Y_query = feat['query']['pids']
    pids = np.unique(Y_gallery)

    X_gallery = X_gallery[Y_gallery != 0]
    Y_gallery = Y_gallery[Y_gallery != 0]

    id = 1501
    gallery_feat = X_gallery[Y_gallery == id][0]
    query_feat_pos = X_query[Y_query == id][0]
    query_feat_neg = X_query[Y_query != id][0]

    # print(np.linalg.norm(gallery_feat - query_feat_pos))
    # print(np.linalg.norm(gallery_feat - query_feat_neg))
    # pass

    # print(jaccard_cont(gallery_feat, query_feat_pos))
    # print(jaccard_cont(gallery_feat, query_feat_neg))
    # print(jaccard_cont(query_feat_pos, query_feat_neg))
    #
    # gallery_feat_torch = torch.Tensor(gallery_feat)
    # query_feat_pos_torch = torch.Tensor(query_feat_pos)
    # query_feat_neg_torch = torch.Tensor(query_feat_neg)
    #
    # print(jaccard_cont_torch(gallery_feat_torch, query_feat_pos_torch))
    # print(jaccard_cont_torch(gallery_feat_torch, query_feat_neg_torch))
    # print(jaccard_cont_torch(query_feat_pos_torch, query_feat_neg_torch))

    #print(jaccard_cont_torch(torch.Tensor(X_gallery), torch.Tensor(X_query)))

    random_indices_gallery = np.random.choice(Y_gallery.shape[0], size=500, replace=False)
    random_indices_query = np.random.choice(Y_query.shape[0], size=500, replace=False)
    compute_pairwise_gallery_query(X_gallery[random_indices_gallery, :], Y_gallery[random_indices_gallery],
                                   X_query[random_indices_query, :], Y_query[random_indices_query])


def consumer1(arg):
    print(arg)


def consumer2(arg):
    print(arg)


def test(**kwargs):
    consumer1(**kwargs)

    print(kwargs)
    if 'arg' in kwargs:
        consumer2(**kwargs)


test(arg=5)