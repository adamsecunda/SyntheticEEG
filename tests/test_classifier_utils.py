import torch
import numpy as np
import pytest
from utils.classifier_utils import DeepConvNet, train_model


class TestDeepConvNet:
    """
    Tests using dummy inputs. No training needed, just checking the
    forward pass produces the right output shape.
    """

    @pytest.fixture(scope="class")
    def model(self):
        return DeepConvNet()

    def test_output_shape(self, model):
        x = torch.zeros(8, 22, 1001)
        assert model(x).shape == (8, 4)

    def test_output_shape_single(self, model):
        x = torch.zeros(1, 22, 1001)
        assert model(x).shape == (1, 4)

    def test_eval_mode(self, model):
        model.eval()
        x = torch.zeros(8, 22, 1001)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (8, 4)


class TestTrainModel:
    """
    Full training on the real dataset is too slow for tests, so we use
    a small notional dataset and run for 5 epochs to verify the function
    runs and returns values of the correct type and shape.
    """

    @pytest.fixture(scope="class")
    def results(self):
        X = np.random.randn(100, 22, 1001).astype("float32")
        y = np.random.randint(0, 4, size=100)
        return train_model(X, y, n_epochs=5, verbose=False)

    def test_best_acc_in_range(self, results):
        best_acc, _ = results
        assert 0 <= best_acc <= 1

    def test_class_accs_length(self, results):
        _, class_accs = results
        assert len(class_accs) == 4

    def test_class_accs_in_range(self, results):
        _, class_accs = results
        assert all(0 <= acc <= 1 for acc in class_accs)