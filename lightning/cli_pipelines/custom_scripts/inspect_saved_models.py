import torch

prism_model_path = '/home/workspace/Ranking-based-Instance-Selection/output/CARSN-BN-MemoryContrastiveLossPRISM-sTRM/model_best.pth'

my_prism_model_path = '/home/workspace/object_classifier_deploy/lightning/cli_pipelines/lightning_logs/cars_LitInception/version_14/checkpoints/epoch-49.ckpt'

prism_model = torch.load(prism_model_path)
my_prism_model = torch.load(my_prism_model_path)

d_orig = prism_model['model']
d_my = my_prism_model['state_dict']

print(len(d_orig), len(d_my))

extra_keys = set(d_orig.keys()) ^ set(d_my.keys())

for k in extra_keys:
    print(k)

pass