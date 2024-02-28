import itertools
import warnings
from typing import Iterator, Tuple, List, Dict
import torch
from vast.opensetAlgos.EVM import set_cover, fit_low
from vast.tools import pairwisedistances


def EVM_Training(
    pos_classes_to_process: List[str],
    features_all_classes: Dict[str, torch.Tensor],
    args,
    gpu: int,
    models=None,
) -> Iterator[Tuple[str, Tuple[str, dict]]]:
    """
    Documentation ins omitted, since it is the same as EVM_training except covered vectors are also returned
    """
    device = "cpu" if gpu == -1 else f"cuda:{gpu}"
    negative_classes_for_current_batch = []
    no_of_negative_classes_for_current_batch = 0
    temp = []
    for cls_name in set(features_all_classes.keys()) - set(pos_classes_to_process):
        no_of_negative_classes_for_current_batch += 1
        temp.append(features_all_classes[cls_name])
        if len(temp) == args.chunk_size:
            negative_classes_for_current_batch.append(torch.cat(temp))
            temp = []
    if len(temp) > 0:
        negative_classes_for_current_batch.append(torch.cat(temp))
    for pos_cls_name in pos_classes_to_process:
        # Find positive class features
        positive_cls_feature = features_all_classes[pos_cls_name].to(device)
        tailsize = max(args.tailsize)
        if tailsize <= 1:
            neg_count = sum(n.shape[0] for c,n in features_all_classes.items() if c != pos_cls_name)
            tailsize = tailsize * neg_count
        tailsize = int(tailsize)

        negative_classes_for_current_class = []
        temp = []
        neg_cls_current_batch = 0
        for cls_name in set(pos_classes_to_process) - {pos_cls_name}:
            neg_cls_current_batch += 1
            temp.append(features_all_classes[cls_name])
            if len(temp) == args.chunk_size:
                negative_classes_for_current_class.append(torch.cat(temp))
                temp = []
        if len(temp) > 0:
            negative_classes_for_current_class.append(torch.cat(temp))
        negative_classes_for_current_class.extend(negative_classes_for_current_batch)

        assert (
            len(negative_classes_for_current_class) >= 1
        ), "In order to train the EVM you need atleast one negative sample for each positive class"
        bottom_k_distances = []
        for batch_no, neg_features in enumerate(negative_classes_for_current_class):
            assert positive_cls_feature.shape[0] != 0 and neg_features.shape[0] != 0, (
                f"Empty tensor encountered positive_cls_feature {positive_cls_feature.shape}"
                f"neg_features {neg_features.shape}"
            )
            distances = pairwisedistances.__dict__[args.distance_metric](
                positive_cls_feature, neg_features.to(device)
            )
            bottom_k_distances.append(distances.cpu())
            bottom_k_distances = torch.cat(bottom_k_distances, dim=1)
            # Store bottom k distances from each batch to the cpu
            bottom_k_distances = [
                torch.topk(
                    bottom_k_distances,
                    min(tailsize, bottom_k_distances.shape[1]),
                    dim=1,
                    largest=False,
                    sorted=True,
                ).values
            ]
            del distances
        bottom_k_distances = bottom_k_distances[0].to(device)

        # Find distances to other samples of same class
        positive_distances = pairwisedistances.__dict__[args.distance_metric](
            positive_cls_feature, positive_cls_feature
        )
        # check if distances to self is zero
        e = torch.eye(positive_distances.shape[0]).type(torch.BoolTensor)
        if not torch.allclose(
            positive_distances[e].type(torch.FloatTensor),
            torch.zeros(positive_distances.shape[0]),
            atol=1e-05,
        ):
            warnings.warn(
                "Distances of samples to themselves is not zero. This may be due to a precision issue or something might be wrong with you distance function."
            )

        for distance_multiplier, cover_threshold, org_tailsize in itertools.product(
            args.distance_multiplier, args.cover_threshold, args.tailsize
        ):
            if org_tailsize <= 1:
                neg_count = sum(n.shape[0] for c,n in features_all_classes.items() if c != pos_cls_name)
                tailsize = int(org_tailsize * neg_count)
            else:
                tailsize = int(org_tailsize)
            # Perform actual EVM training
            weibull_model = fit_low(bottom_k_distances, distance_multiplier, tailsize, gpu)
            # If cover_threshold is greater than 1.0 then do not run set cover rather
            # just consider all samples as extreme vector indices. This is what set cover will do even if it is run.
            if cover_threshold < 1.0:
                (
                    extreme_vectors_models,
                    extreme_vectors_indexes,
                    covered_vectors,
                ) = set_cover(
                    weibull_model, positive_distances.to(device), cover_threshold
                )
                extreme_vectors = torch.gather(
                    positive_cls_feature,
                    0,
                    extreme_vectors_indexes[:, None]
                    .to(device)
                    .repeat(1, positive_cls_feature.shape[1]),
                )
                covered_vectors = list(map(torch.squeeze, covered_vectors))
            else:
                (
                    extreme_vectors_models,
                    extreme_vectors_indexes,
                ) = weibull_model, torch.arange(positive_cls_feature.shape[0])
                extreme_vectors = positive_cls_feature
                covered_vectors = None
            extreme_vectors_models.tocpu()
            extreme_vectors = extreme_vectors.cpu()
            yield (
                f"TS_{org_tailsize}_DM_{distance_multiplier:.2f}_CT_{cover_threshold:.2f}",
                (
                    pos_cls_name,
                    dict(
                        # torch.Tensor -- The extreme vectors used by EVM
                        extreme_vectors=extreme_vectors,
                        # torch.LongTensor -- The index of the above extreme_vectors corresponding to their location in
                        # features_all_classes, only useful if you want to reduce the size of EVM model you save.
                        extreme_vectors_indexes=extreme_vectors_indexes,
                        # weibull.weibull class obj -- the output of weibulls.return_all_parameters() combined with the
                        # extreme_vectors is the actual EVM model for one given class.
                        weibulls=extreme_vectors_models,
                        covered_vectors=covered_vectors,
                    ),
                ),
            )