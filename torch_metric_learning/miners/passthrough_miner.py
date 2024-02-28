from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


class PassThroughMiner(BaseMiner):
    """ A miner which does nothing. Suitable for 'pre-mining' miners as the Population Aware miner is """
    def mine(self, embeddings, labels, ref_emb, ref_labels):
        return lmu.get_all_pairs_indices(labels, ref_labels)

