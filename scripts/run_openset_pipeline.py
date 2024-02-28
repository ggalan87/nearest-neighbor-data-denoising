import os
from evt.vast_openset import *
from evt.vast_ext import filter_labels
from features_storage import FeaturesStorage

# 1. Select dataset features, FeaturesStorage output format, convert them in OpensetData format
features_path = '../lightning/features/mnist_LitModel.pt'
assert os.path.exists(features_path)

fs = FeaturesStorage(cached_path=features_path)

training_feats_dict = fs.training_feats
testing_feats_dict = fs.testing_feats

# 2. Specify labels to be included in training set
traning_class_labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.int32)

training_feats = OpensetData(*filter_labels(traning_class_labels, training_feats_dict['feats'], training_feats_dict['labels']))
testing_feats = OpensetData(testing_feats_dict['feats'], testing_feats_dict['labels'])

# 3. Specify algorithm parameters
approach = 'OpenMax'
algorithm_parameters = "--distance_metric euclidean --tailsize 0.1"
saver_parameters = f"--OOD_Algo {approach}"
model_params = OpensetModelParameters(approach, algorithm_parameters, saver_parameters)

# 4. Run the pipeline
trainer = OpensetTrainer(training_feats, model_params)

trainer.train()
trainer.eval(testing_feats)
trainer.plot(testing_feats)
