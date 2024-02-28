# Original loss
#python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30
#python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --trainer.max_epochs=30

# MS Loss
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 60000}"

# SubCenterArcFaceLoss
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 60000}"

# SoftTripleLoss
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 60000}"

python extract_feats.py --dataset-name NoisyMNIST --model-name LitModel --versions-range 0 15  --epochs-range 0 30 --batch-keys "image" "target" "target_orig" "data_idx" --parts-list "test"
