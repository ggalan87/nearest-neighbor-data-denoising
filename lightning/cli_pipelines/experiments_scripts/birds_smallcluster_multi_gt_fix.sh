# Original loss
python run.py fit --config ./configs/birds/config_birds_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --data.init_args.dataset_args.init_args.training_variant="CUB_0.25smallclusternoised"
python run.py fit --config ./configs/birds/config_birds_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --data.init_args.dataset_args.init_args.training_variant="CUB_0.5smallclusternoised"

# MS Loss
python run.py fit --config ./configs/birds/config_birds_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}" --data.init_args.dataset_args.init_args.training_variant="CUB_0.25smallclusternoised"
python run.py fit --config ./configs/birds/config_birds_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.MultiSimilarityLoss --model.init_args.loss_kwargs="{}" --data.init_args.dataset_args.init_args.training_variant="CUB_0.5smallclusternoised"

# SubCenterArcFaceLoss
python run.py fit --config ./configs/birds/config_birds_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CUB_0.25smallclusternoised"
python run.py fit --config ./configs/birds/config_birds_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SubCenterArcFaceLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CUB_0.5smallclusternoised"

# SoftTripleLoss
python run.py fit --config ./configs/birds/config_birds_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CUB_0.25smallclusternoised"
python run.py fit --config ./configs/birds/config_birds_replicate.yaml --config ./configs/common/noise_reduction/gt_noise_reduction.yaml --trainer.max_epochs=30 --model.init_args.loss_class=pytorch_metric_learning.losses.SoftTripleLoss --model.init_args.loss_kwargs="{'optimizer_kwargs': {'lr': 0.003}}" --data.init_args.dataset_args.init_args.training_variant="CUB_0.5smallclusternoised"


python extract_feats.py --dataset-name Birds --model-name LitInception --versions-range 162 169  --epochs-range 0 30 --batch-keys "image" "target" "target_orig" "data_idx" --parts-list "test"
