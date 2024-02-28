from pytorch_metric_learning.losses import ContrastiveLoss, CrossBatchMemory
from pytorch_metric_learning.distances import CosineSimilarity

loss = ContrastiveLoss()
xbm_loss = CrossBatchMemory(loss, embedding_size=512, memory_size=9500)
