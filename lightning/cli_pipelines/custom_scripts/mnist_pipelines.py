from lightning.models.simple_model import LitModel
from lightning.data.data_modules import get_default_transforms, patch_visiondatamodule
import pl_bolts.datamodules

from pipelines.pipelines import *
from default_settings import *
from lightning.data.samplers import RandomIdentitySampler
import lightning.data.datasets

seed_everything(13)

# PATH_DATASETS = '/content/drive/MyDrive/colab_data/datasets'

# Which dataset
dataset_name = 'MNIST'

# Init DataModule
transforms = get_default_transforms(dataset_name)

# Dynamically construct the class from the class name and the corresponding implementations in pl_bolts
class_ = getattr(pl_bolts.datamodules, f'{dataset_name}DataModule')

# Override with <dataset_name>Ext variant whose __get_item__ method returns batch in dict instead of tuple
class_.dataset_cls = getattr(lightning.data.datasets, f'{dataset_name}Ext')

patch_visiondatamodule(sampler_class=RandomIdentitySampler, num_instances=32)

dm = class_(PATH_DATASETS, batch_size=BATCH_SIZE) #, train_transforms=transforms['train'], val_transforms=['test'], test_transforms=transforms['test'])

# Init model from datamodule's attributes
model = LitModel(*dm.dims, dm.num_classes, learning_rate=2e-4,
                loss_args={'loss_weight': 1.0, 'loss_warm_up_epochs': 5,
                           'semi_hard_warm_up_epochs': 10, 'population_warm_up_epochs': 10}, )

# # ckpt_path = 'lightning_logs/fashion_mnist_LitModel/default/version_14/checkpoints/epoch=15-step=2901.ckpt'
ckpt_path = None
training_pipeline(model, dm, evaluate=True, resume_from=ckpt_path, max_epochs=60, gpus=AVAIL_GPUS)

# Specify model weights path
#weight_path = f'./lightning_logs/{dm.name}_LitModel/default/version_17/checkpoints/epoch=59-step=10877.ckpt'
#model = LitModel.load_from_checkpoint(weight_path)
#
# testing_pipeline(dm, model, extract_features=False, visualize=False, evaluate=True, eval_type='reid')

# checkpoints_template = f'./lightning_logs/{dm.name}_LitModel/default/version_{{}}/checkpoints/epoch=59-step=10877.ckpt'
# checkpoints = [checkpoints_template.format(ver) for ver in range(17, 20)]
# aligned_visualization_pipeline(data_module=dm, checkpoints_paths=checkpoints, model_class=LitModel)

# explain_pipeline(dm, model, show=True)
# model_visualization_pipeline(model, dm, verbose=False)
