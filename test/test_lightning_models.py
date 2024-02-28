from lightning.models.inception import LitInception


class TestLightningModel:
    def test_inception(self):
        try:
            _ = LitInception(
                batch_size=64,
                num_classes=98,
                num_channels=3,
                use_pretrained_weights=True,
                noise_reducer_kwargs={
                    'strategy': 'torch_metric_learning.noise_reducers.sample_rejection.KNNPairRejection',
                    'strategy_kwargs': {
                        'rejection_criteria':
                            'torch_metric_learning.noise_reducers.sample_rejection.HighScoreInPositiveClassCriterion',
                        'rejection_criteria_kwargs': {'threshold': 0.5}
                    },
                    'memory_size': 8096
                },
                loss_kwargs={
                    # 'cross_batch_memory': {
                    #     'memory_size': 8096
                    # }
                },
                # ..
                model_variant='bninception'
            )
        except Exception:
            assert False
