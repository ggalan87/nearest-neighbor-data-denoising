# Generate multiple commands to test if a loss is somewhat durable to random label noise
from itertools import product

loss_configurations = \
    {
        'pytorch_metric_learning.losses.MultiSimilarityLoss': {},
        # 'pytorch_metric_learning.losses.TripletMarginLoss': {'margin': 0.2},
        # 'pytorch_metric_learning.losses.FastAPLoss': {},
        # 'pytorch_metric_learning.losses.NormalizedSoftmaxLoss': {'optimizer_kwargs': {'lr': 3e-03}},
        # 'pytorch_metric_learning.losses.ProxyNCALoss': {'optimizer_kwargs': {'lr': 3e-03}},
        # 'pytorch_metric_learning.losses.ProxyAnchorLoss': {'optimizer_kwargs': {'lr': 3e-03}},
        'pytorch_metric_learning.losses.SubCenterArcFaceLoss': {'optimizer_kwargs': {'lr': 3e-03}},
        # 'pytorch_metric_learning.losses.NTXentLoss': {},
        # 'pytorch_metric_learning.losses.SupConLoss': {},
        'pytorch_metric_learning.losses.SoftTripleLoss': {'optimizer_kwargs': {'lr': 3e-03}},
    }

miner_configurations = \
    {
        'null': {},
        #'pytorch_metric_learning.miners.BatchEasyHardMiner': {}
    }

# noise_reducer_configurations = ['null', 'dummy']
noise_reducer_configurations = \
    {
        'null': {},
        'dummy': {},
        'populations': {'decision_threshold': 0.50, 'with_inspector': True, 'strategy': 'knn', 'training_samples_fraction': 1.0, 'noise_handling_kwargs': {'nn_weights': {'k_neighbors': 200}}}
    }
noise_ratio = 0.5

template = """
python run.py fit --config ./configs/birds/config_birds_replicate.yaml \\
    --model.init_args.loss_class={} \\
    --model.init_args.loss_kwargs=\"{}\" \\
    --model.init_args.miner_class={} \\
    --model.init_args.miner_kwargs=\"{}\" \\
    --model.init_args.noise_reducer={} \\
    --model.init_args.noise_reducer_kwargs=\"{}\" \\
    --data.init_args.val_split=0.0 \\
    --data.init_args.num_workers=7 \\
    --data.init_args.data_dir=/data/datasets \\
    --data.init_args.dataset_args.init_args.training_variant="CUB_{}noised" \\
    --trainer.max_epochs=30
"""

all_configs = list(product(loss_configurations.items(), miner_configurations.items(), noise_reducer_configurations.items()))
for (loss_class, loss_kwargs), (miner_class, miner_kwargs), (noise_reducer, noise_reducer_kwargs) in all_configs:
    print(template.format(loss_class, loss_kwargs, miner_class, miner_kwargs, noise_reducer, noise_reducer_kwargs, noise_ratio))

# for loss_config, miner_config, noise_reducer_config in all_configs:
#     print(f'loss: {loss_config}, triplet miner: {miner_config}, noise reduction: {noise_reducer_config}')
# print(f'Total configs: {len(all_configs)}')


# for loss_config_name in loss_configurations.keys():
#     print('\'{}\','.format(loss_config_name.split('.')[-1]))