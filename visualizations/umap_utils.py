import numpy as np
import umap


def create_identical_relation_dict_same(data_length, n_feats):
    """
    Creates a relation dict for AlignedUMAP where the samples correspond to exactly the same data
    :param data_length: the number of data samples
    :param n_feats: the number of the 'parallel' features for which we want to compute the aligned embedding or
    number of 'models' to compare
    :return: the relations dict
    """
    constant_dict = {i: i for i in range(data_length)}
    constant_relations = [constant_dict for i in range(n_feats - 1)]
    return constant_relations


def compute_embeddings(features_list: list, targets=None, output_path='umap.npy'):
    """
    Computes aligned embeddings from a list of features. Each list element corresponds to features that have been
    extracted from different 'models'. By models we mean different CNNs, or same CNN but different layers
    :param features_list: The list of features
    :param output_path: A path to save the embeddings
    :return: The embeddings that were computed
    """
    constant_relations = create_identical_relation_dict_same(features_list[0].shape[0], n_feats=len(features_list))

    aligned_mapper = umap.AlignedUMAP(verbose=True).fit(features_list, y=targets, n_neighbors=100, relations=constant_relations)

    np.save(output_path, aligned_mapper.embeddings_)
    return aligned_mapper.embeddings_
