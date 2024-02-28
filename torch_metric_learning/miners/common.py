import torch
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


def get_all_pairs_indices(labels, ref_labels=None):
    """
    Copied from lmu, because I override the function by cutting and pasting in lmu module
    """
    matches, diffs = lmu.get_matches_and_diffs(labels, ref_labels)
    a1_idx, p_idx = torch.where(matches)
    a2_idx, n_idx = torch.where(diffs)
    return a1_idx, p_idx, a2_idx, n_idx


def filter_indices_tuple(indices_tuple, batch_noise_predictions, noisy_positive_as_negative=False):
    # Extract indices from tuple
    a1_idx, p_idx, a2_idx, n_idx = indices_tuple

    batch_size = len(batch_noise_predictions)
    for i in range(batch_size):
        # Sample predicted as noisy

        if torch.eq(batch_noise_predictions[i], True):
            # Below we select all positives except the one that we found it is bad positive
            bad_indices = torch.bitwise_or(p_idx == i, a1_idx == i)

            # Consider bad positive samples as negative
            if noisy_positive_as_negative:
                a2_idx = torch.cat((a2_idx, a1_idx[bad_indices]))
                n_idx = torch.cat((n_idx, p_idx[bad_indices]))

            good_indices = torch.bitwise_not(bad_indices)
            a1_idx = a1_idx[good_indices]
            p_idx = p_idx[good_indices]

            # also exclude from negatives because might not be negative for some classes but positive
            excluded_bad = torch.bitwise_not(torch.bitwise_or(n_idx == i, a2_idx == i))
            a2_idx = a2_idx[excluded_bad]
            n_idx = n_idx[excluded_bad]

    return a1_idx, p_idx, a2_idx, n_idx