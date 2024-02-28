import pickle
from pathlib import Path
from lightning.data.datasets.noisy_mnist import obtain_noisy_label_configs
import matplotlib.pyplot as plt
import numpy as np


def plot(bh, pabh, plot_title):
    plt.figure()
    plt.plot(np.array(bh), label='BH Mining')
    plt.plot(np.array(pabh), label='Population Aware BH Mining')
    plt.title(f'accuracy per epoch / {plot_title}')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.xticks(np.arange(0, 10, 1))
    plt.ylim((0.0, 1.0))
    plt.legend()

noisy_configs = obtain_noisy_label_configs(n_classes=10, n_excluded=(1, 2), noisiness_perc=1.0)

dataset_name = 'NoisyMNIST'
dm_name = dataset_name.lower()
model_class_name = 'ResNetMNIST'

root_path = Path(f'./lightning_logs/{dm_name}_{model_class_name}_batch_run')

comparison_results_path = root_path / 'comparison'
comparison_results_path.mkdir(exist_ok=True)

n_configs = len(noisy_configs)
for v in range(n_configs):
    run_path_bh_acc = root_path / f'version_{v}' / 'acc.txt'
    run_path_pabh_acc = root_path / f'version_{v + n_configs}' / 'acc.txt'

    with open(run_path_bh_acc, 'rb') as f:
        bh_acc = pickle.load(f)

    with open(run_path_pabh_acc, 'rb') as f:
        pabh_acc = pickle.load(f)

    title = 'excluded set: {}'.format(','.join(map(str, noisy_configs[v].keys())))
    plot(bh_acc, pabh_acc, title)
    plt.savefig(str(comparison_results_path / f'{title}.png'))
