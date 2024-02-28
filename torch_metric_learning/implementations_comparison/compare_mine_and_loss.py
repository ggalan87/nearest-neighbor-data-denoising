from compare_loss import *
from compare_mining import *
from losses import TripletLoss


def method_ptml(embeddings, labels):
    a, ap, an = pytorch_metric_learning_batch_hard_miner(embeddings, labels)
    loss = pytorch_metric_learning_triplet_margin_loss(embeddings, labels, (a, ap, an))
    return loss


def method_torchreid(embeddings, labels):
    a, ap, an = torchreid_batch_hard_miner(embeddings, labels)
    loss = torchreid_triplet_margin_loss(embeddings, labels, (a, ap, an))
    return loss


embeddings = torch.load('embeddings.pt')
labels = torch.load('labels.pt')

assert method_ptml(embeddings, labels) == method_torchreid(embeddings, labels)
