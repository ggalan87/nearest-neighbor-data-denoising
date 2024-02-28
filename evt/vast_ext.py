import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from typing import Dict
import torch
from vast.opensetAlgos import save_load_operations as slop


def ModelSaver_Params(parser):
    ModelSaver_params = parser.add_argument_group("MultiModalOpenMax params")

    ModelSaver_params.add_argument(
        "--output_dir",
        default="./openset_models",
        type=str,
        help="The directory to store the trained model. default: %(default)s",
    )
    ModelSaver_params.add_argument(
        "--OOD_Algo",
        default="Unknown",
        type=str,
        help="The name of the algorithm. It is used as prefix for the filename. default: %(default)s",
    )

    return parser, dict(
        group_parser=ModelSaver_params,
        param_names=("output_dir", "OOD_Algo"),
        param_id_string="{}_{}",
    )


class ModelSaverFixed(slop.model_saver):
    def process_dict(self, group, model):
        """
        Overrides the original method in order to handle the conversion of weibull class to dict which is missing from
        the implementation

        @param group: A group in terms of hdf5 terminology
        @param model: The model to be saved
        @return: None
        """
        for key_name in model:
            # This is a quick patch. A more proper way is to also restructure the class to directly use the builtin
            # function vars() returned dict rather than renaming the parameters and introduce e.g. a Serializable
            # interface to check the type
            value_can_be_dict = key_name in ['weibull', 'weibulls']

            if type(model[key_name]) == dict or value_can_be_dict:
                sub_group = group.create_group(f"{key_name}")
                self.process_dict(sub_group,
                                  model[key_name].return_all_parameters() if value_can_be_dict else model[key_name])
            else:
                group.create_dataset(f"{key_name}", data=model[key_name])


def plot_probabilities_histogram(true_positive_probabilities: torch.Tensor, true_negative_probabilities: torch.Tensor):
    """
    Plots two overlapping histograms; one for the probabilities of the true positive samples and one for the
    probabilities of the true negative samples. The histograms are normalized separately, i.e. bars of each histogram
    sums to 100%.

    @param true_positive_probabilities: The probabilities of the true positive samples
    @param true_negative_probabilities: The probabilities of the true negative samples
    @return: None
    """
    # the histogram of the data
    plt.figure()

    tpp = true_positive_probabilities.numpy()
    tnp = true_negative_probabilities.numpy()

    n, bins, patches = plt.hist(tpp, 11, weights=np.ones(len(tpp)) / len(tpp), density=False, facecolor='g',
                                alpha=0.5, label='true known')
    n, bins, patches = plt.hist(tnp, 11, weights=np.ones(len(tnp)) / len(tnp), density=False, facecolor='r',
                                alpha=0.5, label='true unknown')
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Probabilities')
    plt.ylabel('Frequencies')
    plt.title('Probability of inclusion')
    plt.legend()
    plt.show()


def filter_labels(accepted_labels: torch.Tensor, features: torch.Tensor, class_labels: torch.Tensor,
                  class_labels_to_names: Dict[int, str] = None):
    """
    Filters input such that data that is kept corresponds to classes that are encountered in accepted_labels parameter

    @param accepted_labels: a tensor that contains only the labels to be kept
    @param features: The features tensor - NxD
    @param class_labels: The class labels tensor - Nx1
    @param class_labels_to_names: The mapping from class labels to names - len=N
    @return: The filtered tensors and mapping dictionary
    """
    if features.shape[0] != class_labels.shape[0]:
        raise AssertionError(f'Length of features is not equal to the length of corresponding labels.')

    # Check that accepted_labels is subset of the given labels
    all_labels = torch.unique(accepted_labels)
    accepted_labels = torch.unique(accepted_labels)  # expectation is that unique labels will be passed
    if not all([l in all_labels for l in accepted_labels]):
        raise AssertionError(
            f'Invalid labels subset. Accepted labels {accepted_labels} should be a subset of all labels'
            f' {all_labels}')

    # First, we initialize labels mask to be equal to all zeros, i.e. no indices selected
    labels_mask = torch.zeros_like(class_labels, dtype=torch.bool)

    # Second, we pass all accepted_labels and we make true the desired indices
    for label in accepted_labels:
        labels_mask[class_labels == label] = True

    if class_labels_to_names is not None:
        filtered_mapping = {k: v for k, v in class_labels_to_names.items() if k in accepted_labels}
    else:
        filtered_mapping = None

    return features[labels_mask], class_labels[labels_mask], filtered_mapping
