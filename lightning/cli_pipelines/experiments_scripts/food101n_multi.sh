# Original loss
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --trainer.max_epochs=30

# MS Loss
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 144086}"

# SubCenterArcFaceLoss
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 144086}"

# SoftTripleLoss
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/no_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --config ./configs/common/noise_reduction/knn_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}"
python run.py fit --config ./configs/food/config_food101n_replicate.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --model.init_args.noise_reducer_kwargs="{'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.PRISM', 'keep_only_good_samples': True, 'memory_size': 144086}"

python extract_feats.py --dataset-name Food101N --model-name LitInception --versions-range 4 19  --epochs-range 0 30 --batch-keys "image" "target" "target_orig" "data_idx" --parts-list "test"
