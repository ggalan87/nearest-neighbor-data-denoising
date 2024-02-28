import pandas as pd
from vast_openset import *
import random

def load_sample_data():
    mnist_training_data = pd.read_csv('./TestData/train_mnist.csv').to_numpy()
    mnist_testing_data = pd.read_csv('./TestData/test_mnist.csv').to_numpy()

    training_features = torch.Tensor(mnist_training_data[:, 1:3])
    testing_features = torch.Tensor(mnist_testing_data[:, 1:3])

    training_labels = torch.IntTensor(mnist_training_data[:, 0])
    testing_labels = torch.IntTensor(mnist_testing_data[:, 0])

    class_labels_to_names = {label.item(): str(label.item()) for label in torch.unique(training_labels)}

    random_indices = \
        torch.randperm(len(training_labels))[:int(0.3 * len(training_labels))]
    training_features = training_features[random_indices]
    training_labels = training_labels[random_indices]

    training_data = OpensetData(training_features, training_labels, dataset_indices=None,
                                class_labels_to_names=class_labels_to_names)
    testing_data = OpensetData(testing_features, testing_labels, dataset_indices=None,
                               class_labels_to_names=class_labels_to_names)
    return training_data, testing_data


training_data, testing_data = load_sample_data()

# Example configurations from https://github.com/Vastlab/vast/blob/main/vast/opensetAlgos/Example.ipynb
example_configurations = \
    [
        ('OpenMax', "--distance_metric euclidean"),
        ('OpenMax', "--distance_metric euclidean --tailsize 0.5"),
        ('OpenMax', "--distance_metric euclidean --tailsize 0.1"),
        ('EVM', "--distance_metric euclidean --distance_multiplier 0.7"),
        ('EVM', "--distance_metric euclidean --tailsize 0.7 --distance_multiplier 0.7"),
        ('EVM', "--distance_metric euclidean --tailsize 0.5 --distance_multiplier 0.7"),
        ('EVM', "--distance_metric euclidean --tailsize 0.1 --distance_multiplier 0.7"),
        ('EVM', "--distance_metric euclidean --tailsize 0.01 --distance_multiplier 0.7"),
        # ('MultiModalOpenMax', "--distance_metric euclidean --tailsize 1.0 --Clustering_Algo finch"),
    ]

for (approach, algorithm_parameters) in example_configurations:
    saver_parameters = f"--OOD_Algo {approach}"
    model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

    trainer = OpensetTrainer(training_data, model_params)

    trainer.train()
    trainer.eval(testing_data)
    # trainer.plot(training_data)
    # trainer.save()
    # trainer.load()
    #trainer.plot(testing_data)
