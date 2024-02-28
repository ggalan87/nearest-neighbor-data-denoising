import argparse
import pandas as pd
import h5py
import torch

from vast import opensetAlgos


def structure_data(df):
    data = (df.groupby('GT').apply(lambda x: list(map(list, zip(x['Features1'], x['Features2'])))).to_dict())
    for k in data:
        data[k] = torch.Tensor(data[k])
    return data


def process_dict(group, model):
    print(f'Creating group {group.name}')
    for key_name in model:
        if type(model[key_name]) == dict:
            sub_group = group.create_group(f"{key_name}")
            process_dict(sub_group, model[key_name])
        else:
            group.create_dataset(f"{key_name}", data=model[key_name])
    return

# Load sample data
mnist_training_data = pd.read_csv('./TestData/train_mnist.csv')
mnist_training_data = structure_data(mnist_training_data)

args_string = "--distance_metric euclidean --tailsize 1.0"

parser = argparse.ArgumentParser()
parser, algo_params = opensetAlgos.MultiModalOpenMax_Params(parser)
args = parser.parse_args(args_string.split())

all_hyper_parameter_models = list(
    opensetAlgos.MultiModalOpenMax_Training(
        pos_classes_to_process=[7],
        features_all_classes=mnist_training_data,
        args=args,
        gpu=0,  # to run on CPU
        models=None)
)

# Assumes that there is only one hyper parameter combination and gets model for that combination
models = dict(list(zip(*all_hyper_parameter_models))[1])

# Save
model_file_name = './openset_models/samples/sample.hdf5'
hf = h5py.File(model_file_name, "w")

cls_name = 7
group = hf.create_group(f"{cls_name}")
models[7]['weibulls'] = models[7]['weibulls'].return_all_parameters()

process_dict(group, models[7])

pass