import pickle
from pathlib import Path
import torch

from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
import matplotlib.pyplot as plt


def to_list(t):
    t_l = []
    for i in range(125):
        t_l.append(t[i*64:64 * (i+1)])

    t_l.append(t[8000:8032])
    return t_l


def inspect_for_epoch(output_dir, epoch=0, batch_size=64):
    output_dir = Path(output_dir)

    predictions_per_batch = torch.load(output_dir / f'all_predictions-{epoch}.pt').cpu()
    labels_per_batch = torch.load(output_dir / f'all_labels-{epoch}.pt').cpu()
    noisy_indices_per_batch = torch.load(output_dir / f'all_noisy_indices-{epoch}.pt').cpu()
    labels_to_indices = pickle.load(open(output_dir / f'labels_to_indices-{epoch}.pkl', 'rb'))

    predictions_per_batch = to_list(predictions_per_batch)
    labels_per_batch = to_list(labels_per_batch)
    noisy_indices_per_batch = to_list(noisy_indices_per_batch)

    # n_batches = len(labels_per_batch)

    clean_weights_list = []
    noisy_weights_list = []

    for idx, (preds, labels, noisy_indices) in enumerate(zip(predictions_per_batch, labels_per_batch, noisy_indices_per_batch)):
        a1_idx, p_idx, a2_idx, n_idx = lmu.get_all_pairs_indices(labels)

        model_indices_t = torch.tensor(list(map(lambda l: labels_to_indices[int(l)], labels)))

        w_a1 = preds[a1_idx.cpu(), model_indices_t[a1_idx.cpu()]]
        w_p = preds[p_idx.cpu(), model_indices_t[p_idx.cpu()]]

        w_a2 = preds[a2_idx.cpu(), model_indices_t[a2_idx.cpu()]]
        w_n = preds[n_idx.cpu(), model_indices_t[a2_idx.cpu()]]

        # gt_w_a1 = torch.ones_like(a1_idx) - noisy_indices_per_batch[a1_idx]
        # gt_w_p = torch.ones_like(p_idx) - noisy_indices_per_batch[p_idx]
        # gt_w_a2 = torch.ones_like(a2_idx) - noisy_indices_per_batch[a2_idx]
        # gt_w_n = torch.ones_like(n_idx) - noisy_indices_per_batch[n_idx]

        n = len(labels)
        weights_matrix = torch.ones(n, n, device=preds.device, dtype=torch.float64)

        anchor_idx, positive_idx, negative_idx = lmu.convert_to_triplets(
            (a1_idx, p_idx, a2_idx, n_idx), labels, ref_labels=None, t_per_anchor='all'
        )

        ap_weights = preds[anchor_idx, positive_idx]
        n_weights = preds[anchor_idx, negative_idx]

        # ap_weights = weights_matrix[anchor_idx, positive_idx]
        # n_weights = weights_matrix[anchor_idx, negative_idx]

        batch_weights = ap_weights * n_weights

        # Triplet is noisy if either sample is noisy
        triplet_noisy_indices = noisy_indices[anchor_idx] | noisy_indices[positive_idx] | noisy_indices[negative_idx]
        clean_weights = batch_weights[triplet_noisy_indices == False]
        noisy_weights = batch_weights[triplet_noisy_indices == True]

        clean_weights_list.append(clean_weights)
        noisy_weights_list.append(noisy_weights)

        if idx == 0:
            plt.matshow(preds.numpy(), vmin=0, vmax=1)
            plt.colorbar()
            plt.savefig(f'{output_dir}/epoch-{epoch}-batch-{idx}.png')


    all_clean_weights = torch.cat(clean_weights_list)
    all_noisy_weights = torch.cat(noisy_weights_list)

    print('Noisy weights: mean: {}'.format(torch.mean(all_noisy_weights)))
    print('Clean weights: mean: {}'.format(torch.mean(all_clean_weights)))


    # mmax = max(all_clean_weights.max(), all_noisy_weights.max())
    # hist_clean = torch.histc(all_clean_weights, bins=20, min=0, max=mmax)
    # hist_noisy = torch.histc(all_noisy_weights, bins=20, min=0, max=mmax)
    #
    # # x = range(mmax/100)
    # # plt.bar(x, hist_clean, align='center', alpha=0.5)
    # # plt.bar(x, hist_noisy, align='center', alpha=0.5)
    # # plt.xlabel('Bins')
    #
    # pass


output_dir = '/home/workspace/object_classifier_deploy/lightning/cli_pipelines/rejection_inspector_output'
for i in range(30):
    inspect_for_epoch(output_dir=output_dir, epoch=i, batch_size=64)



