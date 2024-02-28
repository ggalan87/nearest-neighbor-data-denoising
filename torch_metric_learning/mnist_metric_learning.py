import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_lightning import seed_everything
from lightning.models.simple_model import LitModel
from pytorch_metric_learning.miners import UniformHistogramMiner
from pytorch_metric_learning.distances import SNRDistance
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from torch_metric_learning.miners.population_aware import create_pa_miner
from losses import TripletLoss

seed_everything(13)

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from torchvision import datasets, transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from lightning.data.datasets import DirtyMNIST

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, -1#mining_func.num_triplets
                )
            )

    # if isinstance(mining_func, PopulationAwareMiner):
    if type(mining_func).__name__ == 'PopulationAwareMiner':
        mining_func.bootstrap_epoch(epoch)


def my_train(model, loss_func, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        loss = loss_func(embeddings, labels)
        loss.backward()
        optimizer.step()
        if batch_idx % 20 == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, -1  # mining_func.num_triplets
                )
            )

### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator, epoch=None):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, test_labels, train_embeddings, train_labels
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

    torch.save(train_embeddings, f'./embeddings/train_embeddings_e{epoch}.pt')
    torch.save(train_labels, f'./embeddings/train_labels_e{epoch}.pt')
    torch.save(test_embeddings, f'./embeddings/test_embeddings_e{epoch}.pt')
    torch.save(test_labels, f'./embeddings/test_labels_e{epoch}.pt')


device = torch.device("cuda")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

batch_size = 256

dataset1 = datasets.MNIST(".", train=True, download=True, transform=transform)
dataset2 = datasets.MNIST(".", train=False, transform=transform)

# dataset1 = DirtyMNIST(".", train=True, download=True, transform=transform, dirty_probability=1.0)
# dataset2 = DirtyMNIST(".", train=False, transform=transform, dirty_probability=1.0)

train_loader = torch.utils.data.DataLoader(
    dataset1, batch_size=batch_size, shuffle=True
)
test_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size)

model = Net().to(device)

# model = LitModel(1, 28, 28, 10, learning_rate=2e-4,
#                 loss_args={'loss_weight': 1.0, 'loss_warm_up_epochs': 5,
#                            'semi_hard_warm_up_epochs': 10, 'population_warm_up_epochs': 10}, )
# model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 2


### pytorch-metric-learning stuff ###
distance = distances.LpDistance(normalize_embeddings=True)
reducer = reducers.ThresholdReducer(low=0)
loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
# mining_func = miners.TripletMarginMiner(
#     margin=0.2, distance=distance, type_of_triplets="hard"
# )
mining_func = miners.BatchHardMiner(
    distance=distance
)

# mining_func = create_pa_miner(miners.BatchHardMiner, distance=distance, distance_kwargs={'normalize_embeddings': True})
# mining_func = UniformHistogramMiner(
#     num_bins=100,
#     pos_per_bin=25,
#     neg_per_bin=33,
#     distance=SNRDistance(),
# )

# loss_func = TripletLoss(margin=0.2)

accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
### pytorch-metric-learning stuff ###


for epoch in range(1, num_epochs + 1):
    train(model, loss_func, mining_func, device, train_loader, optimizer, epoch)
    #my_train(model, loss_func, device, train_loader, optimizer, epoch)
    test(dataset1, dataset2, model, accuracy_calculator, epoch)
