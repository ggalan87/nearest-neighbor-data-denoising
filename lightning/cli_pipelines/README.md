# Examples

## run

### Products with resnet and soft triple loss
```
python run fit --config ./configs/online_products/config_online_products.yaml --model ./configs/common/resnet_model_softtriple.yaml --data ./configs/online_products/data_online_products.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="SOP_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.reduced_dim=128 --model.init_args.model_variant="resnet50" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets
```

### FOOD101N with resnet and soft triple loss
```
python run fit --config ./configs/food/config_food101n.yaml --model ./configs/common/resnet_model_softtriple.yaml --data ./configs/food/data_food101n.yaml --data.init_args.val_split=0.0 --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.reduced_dim=128 --model.init_args.model_variant="resnet50" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets
```

### Cars with inception and soft triple loss
```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.model_variant="bninception" --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.35}" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets
```

```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=populations --model.init_args.model_variant="bninception" --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.30, 'use_raw_probabilities': True, 'keep_only_good_samples': True, 'training_samples_fraction': 1.0}" --data.init_args.data_dir=/data/datasets --sampler ./configs/common/random_sampler.yaml
```

### Cars with rest losses
```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.model_variant="bninception" --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.35}" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.FastAPLoss --model.init_args.loss_kwargs="{}"
```

```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.model_variant="bninception" --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.35}" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.NormalizedSoftmaxLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 1e-4}}"
```

```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.model_variant="bninception" --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.35}" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.ProxyNCALoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 1e-4}}"
```

```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.model_variant="bninception" --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.35}" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.ProxyAnchorLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 1e-4}}"
```

```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.model_variant="bninception" --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.35}" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 1e-4}}"
```

```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=50 --model.init_args.noise_reducer=null --model.init_args.model_variant="bninception" --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.35}" --sampler=./configs/common/random_sampler.yaml --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.CrossBatchMemory --model.init_args.loss_kwargs="{'loss': pytorch_metric_learning.losses.ContrastiveLoss, 'memory_size': 8032}"
```

#### Triplet loss
```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=30 --model.init_args.model_variant="bninception" --model.init_args.noise_reducer=null --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.TripletMarginLoss --model.init_args.loss_kwargs="{'margin': 0.2}" --model.init_args.miner_class=pytorch_metric_learning.miners.BatchEasyHardMiner  --sampler.init_args.fix_samples=True
```

```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=30 --model.init_args.model_variant="bninception" --model.init_args.noise_reducer=populations --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.30, 'use_raw_probabilities': False, 'keep_only_good_samples': False}" --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.TripletMarginLoss --model.init_args.loss_kwargs="{'margin': 0.2}" --model.init_args.miner_class=pytorch_metric_learning.miners.BatchEasyHardMiner --sampler.init_args.fix_samples=True
```

```
python run.py fit --config ./configs/cars/config_cars.yaml --model ./configs/common/inception_model_softtriple.yaml --data ./configs/cars/data_cars.yaml --data.init_args.val_split=0.0 --data.init_args.dataset_args.init_args.training_variant="CARS_0.5noised" --trainer.max_epochs=30 --model.init_args.model_variant="bninception" --model.init_args.noise_reducer=populations --model.init_args.noise_reducer_kwargs="{'use_pretrained': True, 'tail_size': 0.30, 'use_raw_probabilities': True, 'keep_only_good_samples': False}" --data.init_args.data_dir=/data/datasets --model.init_args.loss_class=pytorch_metric_learning.losses.TripletMarginLoss --model.init_args.loss_kwargs="{'margin': 0.2}" --model.init_args.miner_class=pytorch_metric_learning.miners.BatchEasyHardMiner --sampler.init_args.fix_samples=True
```