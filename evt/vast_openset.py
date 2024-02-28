import os
import argparse
from typing import Optional, Tuple, Union, Any
from warnings import warn
from torchmetrics.classification import MulticlassAccuracy, BinaryAccuracy, BinaryConfusionMatrix, \
    MulticlassConfusionMatrix, BinaryF1Score
from sklearn.metrics import ConfusionMatrixDisplay
from vast import opensetAlgos
from vast.tools import viz

from lightning.ext import logger
from evt.vast_ext import *
from evt import vast_core as opensetAlgosPlus


class OpensetData:
    def __init__(self, features: torch.Tensor, class_labels: torch.Tensor, dataset_indices: torch.Tensor = None,
                 class_labels_to_names: Dict[int, str] = None):
        """
        @param features: a tensor that holds the features of shape (N, D), where N is the number of samples and D the
        dimensionality of the features
        @param class_labels: a tensor that holds the labels for each sample of shape (N,1), where N is the number of samples
        @param class_labels_to_names: a dictionary which maps class labels to class names
        """
        assert features.shape[0] == class_labels.shape[0]

        unique_labels = torch.unique(class_labels)
        if class_labels_to_names is None:
            class_labels_to_names = {label.item(): str(label.item()).zfill(10) for label in unique_labels}

        self._features_dict, self._indices_dict, self._labels_to_indices \
            = self._convert_data(features, class_labels, dataset_indices, class_labels_to_names)

    def _convert_data(self, features: torch.Tensor, class_labels: torch.Tensor, dataset_indices: torch.Tensor,
                      class_labels_to_names: Dict[int, str]):
        """
        Converts the data to an intermediate format, that is a dictionary with keys the class names and values a list of
        tensors which correspond to the features of the samples

        @param features: NxD features
        @param class_labels: Nx1 class labels
        @param class_labels_to_names: mapping from class labels to class names
        @return:
        """
        features_dict = {}
        indices_dict = {}

        # Find unique names
        unique_class_labels = torch.unique(class_labels)
        for label in unique_class_labels:
            features_dict[class_labels_to_names[label.item()]] = features[class_labels == label]

        if dataset_indices is not None:
            for label in unique_class_labels:
                indices_dict[class_labels_to_names[label.item()]] = dataset_indices[class_labels == label]

        labels_to_indices = {}
        for i, label in enumerate(unique_class_labels):
            labels_to_indices[int(label)] = i

        return features_dict, indices_dict, labels_to_indices

    def back_convert_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        labels_list = []
        for label, feats in self._features_dict.items():
            l = int(label)
            labels_list.append(torch.ones((len(feats),), dtype=torch.int64) * l)

        return torch.cat(list(self._features_dict.values())), torch.cat(labels_list)

    @property
    def features_dict(self):
        return self._features_dict

    @property
    def indices_dict(self):
        return self._indices_dict

    @property
    def labels_to_indices(self) -> Dict[int, int]:
        """
        Indices 0..C-1. Useful for the case that class labels are not in this range
        @return:
        """
        return self._labels_to_indices

    @property
    def features_dimension(self):
        """
        The dimensionality of the features of the dataset. Since it is the same for all we get the dimensionality of the
        first

        @return: the dimensionality of the features
        """
        first_feat = next(iter(self._features_dict.values()))
        return first_feat.shape[1]


class OpensetModelParameters:
    def __init__(self, name, algorithm_parameters, saver_parameters=None):
        # Getting functions to execute
        self._param_fn = getattr(opensetAlgos, f'{name}_Params')

        try:
            self._training_fn = getattr(opensetAlgosPlus, f'{name}_Training')
        except AttributeError:
            self._training_fn = getattr(opensetAlgos, f'{name}_Training')

        self._testing_fn = getattr(opensetAlgos, f'{name}_Inference')

        algo_parser = argparse.ArgumentParser()
        algo_parser, algo_params = self._param_fn(algo_parser)
        self._algo_args = algo_parser.parse_args(algorithm_parameters.split())

        params_values = []
        for param_name in algo_params['param_names']:
            val = getattr(self._algo_args, param_name)
            if isinstance(val, list):
                assert len(val) == 1, "For now we support only single parameter."
                val = val[0]
            params_values.append(val)

        self._hyperparameters_info = algo_params['param_id_string'].format(*params_values)

        if saver_parameters is not None:
            saver_parser = argparse.ArgumentParser()
            saver_parser, saver_params = ModelSaver_Params(saver_parser)
            self._saver_args = saver_parser.parse_args(saver_parameters.split())
        else:
            self._saver_args = None

    @property
    def algo_args(self):
        return self._algo_args

    @property
    def saver_args(self):
        return self._saver_args

    @property
    def training_function(self):
        return self._training_fn

    @property
    def testing_function(self):
        return self._testing_fn

    @property
    def hyperparameters_info(self) -> str:
        return self._hyperparameters_info


class OpensetTrainer:
    def __init__(self, data: OpensetData, model_parameters: OpensetModelParameters, inference_threshold=0.5):
        """

        @param data:
        @param model_parameters:
        @param inference_threshold: The threshold above which a sample will be considered as included
        """
        self._data = data
        self._model_parameters = model_parameters
        self._models = None
        self._threshold = inference_threshold

        self._accuracy_metric = BinaryAccuracy()
        self._confusion_matrix_metric = BinaryConfusionMatrix(normalize='true')
        self._multiclass_confusion_matrix_metric = \
            MulticlassConfusionMatrix(num_classes=len(data.features_dict), normalize='true')
        self._f1_metric = BinaryF1Score()

        self._logger = logger

        # Hooks which take
        self._hooks = \
            {
                'before_training': [],
                'after_training': []
            }

    def train(self):
        self._call_hooks('before_training')

        # Run the training function
        all_hyper_parameter_models = list(
            self._model_parameters.training_function(
                pos_classes_to_process=self._data.features_dict.keys(),
                features_all_classes=self._data.features_dict,
                args=self._model_parameters.algo_args,
                gpu=self.__class__.get_gpu(),
                models=None)
        )

        # Assumes that there is only one hyper parameter combination and gets model for that combination
        self._models = dict(list(zip(*all_hyper_parameter_models))[1])

        # if self._noise_handling_options is not None and \
        #         self._noise_handling_options.density_mr is not None:
        #     self._reduce_models()
        # else:
        #     for label, model in self._models.items():
        #         n_weibs = len(model['extreme_vectors'])
        #         class_features = self._data.features_dict[label]
        #         self._logger.debug(
        #             f'Class {label}, computed {n_weibs} from #{len(class_features)} samples')

        self._call_hooks('after_training')

    def save(self):
        model_saver = ModelSaverFixed(self._model_parameters.saver_args,
                                      process_combination=(self._model_parameters.hyperparameters_info, 0),
                                      total_no_of_classes=len(self._data.features_dict), output_file_name=None)

        for cid in self._models.keys():
            model_saver(cid, self._models[cid])

    def load(self):
        self._models = slop.model_loader(self._model_parameters.saver_args, self._model_parameters.hyperparameters_info)

    def register_hook(self, hook_name, hook_object):
        # TODO: Check that hook to be registered takes single argument which is οφ type OpensetTrainer

        if hook_name not in self._hooks:
            available_hooks = ','.join(list(self._hooks.keys()))
            raise ValueError(f'Invalid hook name. Must be one from: {available_hooks}')
        self._hooks[hook_name].append(hook_object)

    def _call_hooks(self, hook_name):
        for hook in self._hooks[hook_name]:
            hook(self)

    def plot(self, data: OpensetData):
        filename = os.path.join(self._model_parameters.saver_args.output_dir,
                                '{}_{}.pdf'.format(self._model_parameters.saver_args.OOD_Algo,
                                                   self._model_parameters.hyperparameters_info))

        self.__class__._plot_heatmap(data.features_dict, heat_map_fn=self._get_probs, file_name=filename)

        feats, labels, indices = self._obtain_features_and_inclusion_labels(data)
        self._plot_histogram(feats, labels)

    def _get_probs(self, pnts, single_probability=True):
        def ensure_tensor(var):
            if isinstance(var, np.ndarray):
                return torch.from_numpy(var)
            else:
                return var

        pnts = ensure_tensor(pnts)

        # TODO: need to accelerate the testing function due to the fact that it's design is for evaluation and not for \
        #  simple fast inference. This will permit to avoid following conversions of the results and misleading \
        #  arguments that are used abusively, e.g. pos_classes_to_process, features_all_classes
        result = self._model_parameters.testing_function(pos_classes_to_process=('0'),
                                                         features_all_classes={'0': pnts.clone().detach().double()},
                                                         args=self._model_parameters.algo_args,
                                                         gpu=self.__class__.get_gpu(),
                                                         models=self._models)

        # In python 3.7+ dictionary iteration order is guaranteed to be in order of insertion, therefore the indices are
        # labels are 1-1 to class ids (if class ids start from 0 and go up)
        result = torch.cat(list(dict(list(zip(*result))[1]).values()))

        if single_probability:
            result = torch.max(result, dim=1).values

        return result

    def get_inclusion_probabilities(self, data: torch.Tensor):
        """
        It provides per-class inclusion probability in a vector (per sample). The vector is therefore of dimensionality
        C (as the number of known classes), and the content is 0-1 probability.

        @param data: A set of NxD features
        @return: NxC inclusion probabilities, C probabilities for each of N samples
        """
        inclusion_probabilities = self._get_probs(data, single_probability=False)
        return inclusion_probabilities

    def predict(self, data: torch.Tensor, threshold: Optional[float] = None):
        """

        @param data: A set of NxD features
        @param threshold: A threshold which overrides the class member threshold
        @return: Inclusion labels
        """
        inclusion_probabilities = self._get_probs(data)
        predicted_as_included = (inclusion_probabilities >= (self._threshold if threshold is None else threshold)).int()
        return predicted_as_included

    def predict_per_class(self, data: torch.Tensor, threshold: Optional[float] = None):
        """
        Instead of getting the maximum prediction of all classes, it provides per-class prediction and inclusion
        labeling in a vector (per sample). The vector is therefore of dimensionality C (as the number of known classes),
        and the content is 1 for inclusion in the class or 0 for non-inclusion.

        @param data: A set of NxD features
        @param threshold: A threshold which overrides the class member threshold
        @return: NxC inclusion labels, C labels for each of N samples
        """
        inclusion_probabilities = self._get_probs(data, single_probability=False)
        predicted_as_included = (inclusion_probabilities >= (self._threshold if threshold is None else threshold)).int()
        return predicted_as_included

    def predict_classes(self, data: torch.Tensor, threshold: Optional[float] = None, with_noclass=False):
        """
        Attention!. It returns indices for classes 0...C-1

        @param data: Features
        @param threshold: Acceptance threshold
        @param with_noclass: Use an artificial class "noclass" if threshold is not surpassed
        @return: Returns the indices of the classes. If class labels are 0...C-1 then it correctly corresponds to them
        """
        inclusion_probabilities = self._get_probs(data, single_probability=False)
        m = torch.max(inclusion_probabilities, dim=1)
        values, indices = m.values, m.indices

        if with_noclass:
            # assign a label outside the true classes to represent the novel class
            indices[values < (self._threshold if threshold is None else threshold)] = len(self._models)

        return indices

    def predict_proba(self, data: torch.Tensor):
        inclusion_probabilities = self._get_probs(data, single_probability=False)
        return inclusion_probabilities.numpy()

    def eval(self, data: OpensetData, get_metrics=False) -> Any:
        return self._eval(*self._obtain_features_and_inclusion_labels(data), get_metrics=get_metrics)

    def _eval(self, pnts: torch.Tensor, inclusion_labels: torch.Tensor, dataset_indices: torch.Tensor = None,
              get_metrics=False) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict]]:
        inclusion_probabilities = self._get_probs(pnts)
        predicted_as_included = (inclusion_probabilities >= self._threshold).int()
        acc = self._accuracy_metric(predicted_as_included, inclusion_labels)
        confusion_matrix = self._confusion_matrix_metric(predicted_as_included, inclusion_labels)

        f1 = self._f1_metric(predicted_as_included, inclusion_labels)

        if get_metrics:
            metrics = \
                {
                    'acc': acc.item(),
                    'f1': f1.item(),
                    'confusion': confusion_matrix
                }
            return predicted_as_included, dataset_indices, metrics

        print('Accuracy (correct included):  {:.3f}'.format(acc.item()))
        print('F1-score: {:.3f}'.format(f1.item()))
        print('Confusion:\n', confusion_matrix)

        # self._plot_confusion_matrix(confusion_matrix.numpy())
        # plt.show()

        return predicted_as_included, dataset_indices

    def eval_classification(self, data: OpensetData, with_noclass=False) -> torch.Tensor:
        """
        Evaluates classifier in a normal way, to the class of maximum probability. If 'with_noclass' is
        enabled then if no class surpasses the threshold, sample is classified to 'noclass' class

        @param with_noclass:
        @return:
        """

        feats, true_class_labels = data.back_convert_data()
        predicted_class_labels = self.predict_classes(feats, with_noclass=with_noclass, threshold=0.9)

        # Convert class labels to 0...C-1, assume ascending order
        labels_mapping = {}
        unique_labels = torch.unique(true_class_labels)

        for i, l in enumerate(unique_labels):
            labels_mapping[int(l)] = i

        for orig_l, new_l in labels_mapping.items():
            true_class_labels[true_class_labels == orig_l] = new_l

        accuracy = MulticlassAccuracy(num_classes=len(unique_labels))
        acc = accuracy(predicted_class_labels, true_class_labels)
        print('Accuracy {}'.format(acc.item()))
        num_classes = len(data.features_dict)

        if with_noclass:
            num_classes += 1

        confusion_matrix = self._multiclass_confusion_matrix_metric(predicted_class_labels, true_class_labels)
        # print('Confusion:\n', confusion_matrix)

    def _obtain_features_and_inclusion_labels(self, data: OpensetData) -> (torch.Tensor, torch.Tensor):
        """
        Iterates the features dict (class_label -> features) of the given data set and finds which of the labels were
        present in the training set. These labels are considered as the known classes (inclusion label = one), while the
        rest as unknown (inclusion label = zero).

        @param data: The data to annotate as known vs unknown
        @return: The raw NxD features of the dataset and an Nx1 indicator of knowns vs unknowns (inclusion labels)
        """
        feats_list = []
        inclusion_labels_list = []
        dataset_indices_list = []

        for k, v in data.features_dict.items():
            feats_list.append(v)
            label_fun = torch.ones if k in self._data.features_dict else torch.zeros
            inclusion_labels_list.append(label_fun(v.shape[0], dtype=torch.int32))

        for k, v in data.indices_dict.items():
            dataset_indices_list.append(v)

        feats = torch.cat(feats_list, dim=0)
        inclusion_labels = torch.cat(inclusion_labels_list, dim=0)
        dataset_indices = torch.cat(dataset_indices_list, dim=0) if len(dataset_indices_list) > 0 else None

        return feats, inclusion_labels, dataset_indices

    def _plot_histogram(self, features, inclusion_labels):
        inclusion_probabilities = self._get_probs(features)
        true_positive_probabilities = inclusion_probabilities[inclusion_labels == 1]
        true_negative_probabilities = inclusion_probabilities[inclusion_labels == 0]

        plot_probabilities_histogram(true_positive_probabilities, true_negative_probabilities)

    def _plot_confusion_matrix(self, confusion_matrix):
        ConfusionMatrixDisplay(confusion_matrix, display_labels=['novel', 'existing']).plot()

    @staticmethod
    def _plot_heatmap(dict_data, heat_map_fn=None, *args, **kwargs):
        data = torch.cat(list(dict_data.values()), dim=0)

        if data.shape[1] == 2:
            labels = []
            for k in dict_data:
                labels.extend([k] * dict_data[k].shape[0])
            labels = np.array(labels)
            # raise NotImplementedError('Must fix vastlab deprecated np.float')
            viz.plotter_2D(data.numpy(), labels, final=True,
                           heat_map=heat_map_fn is not None,
                           prob_function=heat_map_fn, *args, **kwargs)
        else:
            warn('Cannot plot a heatmap for more than 2 dimensions.')

    @staticmethod
    def get_gpu():
        device = torch.cuda.current_device() if torch.cuda.is_available() else -1,  # -1 means to run on CPU
        return device[0]

    @property
    def data(self) -> OpensetData:
        return self._data

    @property
    def models(self):
        return self._models
