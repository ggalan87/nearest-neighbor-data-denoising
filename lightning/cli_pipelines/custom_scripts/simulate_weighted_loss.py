from pytorch_metric_learning.losses.triplet_margin_loss import TripletMarginLoss
import torch
from itertools import product

pincl_options_list = [
    {
        'clean': 1.0,
        'noisy': 0.0
    },
    {
        'clean': 0.9,
        'noisy': 0.1
    },
    {
        'clean': 0.4,
        'noisy': 0.6
    },
    {
        'clean': 0.1,
        'noisy': 0.9
    },
    {
        'clean': 0.0,
        'noisy': 1.0
    },
    # {
    #     'clean': 0.9,
    #     'noisy': 0.9
    # },
    # {
    #     'clean': 0.6,
    #     'noisy': 0.6
    # },
    # {
    #     'clean': 0.3,
    #     'noisy': 0.3
    # },
    # {
    #     'clean': 0.1,
    #     'noisy': 0.1
    # }
]
margin = 0.2

positive_distances = torch.normal(mean=torch.tensor(margin), std=torch.tensor(0.2), size=(50, ))
negative_distances = torch.normal(mean=torch.tensor(margin), std=torch.tensor(0.2), size=(50, ))

positive_distances += torch.abs(positive_distances.min())
negative_distances += torch.abs(negative_distances.min())

for pincl_options in pincl_options_list:
    for pd, nd, (ak, av), (pk, pv), (nk, nv) in product(positive_distances, negative_distances, pincl_options.items(),
                                                        pincl_options.items(), pincl_options.items()):
        d_ap = pd
        d_an = nd
        w_ap = av*pv
        nv = 1 - nv
        # w_an = 1 + av*nv
        w_an = av * (nv - 1)

        loss = torch.maximum(torch.tensor([0]), torch.tensor([d_ap - d_an + margin]))
        w_loss = torch.maximum(torch.tensor([0]), torch.tensor([w_ap*d_ap - w_an*d_an + margin]))

        if loss == 0:
            continue

        if w_loss > loss:
            print(f'anchor: {ak}, positive: {pk}, negative: {nk}\nloss: {loss}\nweighted loss: {w_loss}')
            print(f'w_ap: {w_ap}, w_an: {w_an}')
            print(f'd_ap: {d_ap}, d_an: {d_an}')
            print(f'av: {av}, pv: {pv}, nv: {nv}')

            assert False




