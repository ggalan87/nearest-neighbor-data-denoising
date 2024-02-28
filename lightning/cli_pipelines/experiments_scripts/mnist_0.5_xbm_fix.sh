# Original loss
python run.py fit --config ./configs/mnist/config_noisymnist.yaml --config ./configs/common/noise_reduction/prism_nr0.5_noise_reduction.yaml --trainer.max_epochs=30


python extract_feats.py --dataset-name NoisyMNIST --model-name LitModel --versions-list 16  --epochs-range 0 30 --batch-keys "image" "target" "target_orig" "data_idx" --parts-list "test"
