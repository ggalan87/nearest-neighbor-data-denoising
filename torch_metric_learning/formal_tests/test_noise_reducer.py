import torch
from torch_metric_learning.noise_reducers.noise_reducers import DefaultNoiseReducer
from torch_metric_learning.noise_reducers.sample_rejection.dummy import DummyNoiseRejection


class MockData:
    def __init__(self):
        self._batch_size = 64
        self._embedding_size = 128
        self._n_classes = 10

        self._mock_embeddings = torch.rand(self._batch_size, self._embedding_size)
        self._mock_labels = torch.randint(low=0, high=self._n_classes, size=(self._batch_size, ))
        self._noisy_samples = torch.randint(low=0, high=2, dtype=torch.bool, size=(self._batch_size, ))

    @property
    def mock_embeddings(self):
        return self._mock_embeddings

    @property
    def mock_labels(self):
        return self._mock_labels

    @property
    def noisy_samples(self):
        return self._noisy_samples


class TestNoiseReducerDummy:

    def test_without_pretrained(self):
        mock_data = MockData()
        noise_reducer = DefaultNoiseReducer(strategy=DummyNoiseRejection(), use_pretrained=False)

        assert not noise_reducer.use_pretrained

        assert not noise_reducer.has_trained()

        # A first batch
        _ = noise_reducer(mock_data.mock_embeddings, mock_data.mock_labels, mock_data.noisy_samples)
        # A second batch
        batch_noisy_predictions = noise_reducer(mock_data.mock_embeddings, mock_data.mock_labels,
                                                mock_data.noisy_samples)

        assert not torch.any(batch_noisy_predictions)

        noise_reducer.bootstrap_epoch(epoch=1)

        assert noise_reducer.has_trained()

        batch_noisy_predictions = noise_reducer(mock_data.mock_embeddings, mock_data.mock_labels,
                                                mock_data.noisy_samples)

        assert torch.equal(mock_data.noisy_samples, batch_noisy_predictions)

    def test_without_pretrained_and_warmup(self):
        mock_data = MockData()
        noise_reducer = DefaultNoiseReducer(strategy=DummyNoiseRejection(), use_pretrained=False, warm_up_epochs=1)

        assert not noise_reducer.use_pretrained

        assert not noise_reducer.has_trained()

        # A first batch
        batch_noisy_predictions = noise_reducer(mock_data.mock_embeddings, mock_data.mock_labels,
                                                mock_data.noisy_samples)
        assert not torch.any(batch_noisy_predictions)

        noise_reducer.bootstrap_epoch(epoch=1)

        assert not noise_reducer.has_trained()

        batch_noisy_predictions = noise_reducer(mock_data.mock_embeddings, mock_data.mock_labels,
                                                mock_data.noisy_samples)
        assert not torch.any(batch_noisy_predictions)

        noise_reducer.bootstrap_epoch(epoch=2)

        assert noise_reducer.has_trained()

        batch_noisy_predictions = noise_reducer(mock_data.mock_embeddings, mock_data.mock_labels,
                                                mock_data.noisy_samples)

        assert torch.equal(mock_data.noisy_samples, batch_noisy_predictions)

    def test_with_pretrained(self):
        mock_data = MockData()
        noise_reducer = DefaultNoiseReducer(strategy=DummyNoiseRejection(), use_pretrained=True)

        assert noise_reducer.use_pretrained

        noise_reducer.bootstrap_initial(mock_data.mock_embeddings, mock_data.mock_labels)

        assert noise_reducer.has_trained()

        batch_noisy_predictions = noise_reducer(mock_data.mock_embeddings, mock_data.mock_labels,
                                                mock_data.noisy_samples)
        assert torch.equal(mock_data.noisy_samples, batch_noisy_predictions)

    def test_with_pretrained_and_warmup(self):
        try:
            # The code below should throw Value Error
            _ = DefaultNoiseReducer(strategy=DummyNoiseRejection(), use_pretrained=True, warm_up_epochs=1)
        except ValueError:
            pass
        else:
            assert False
