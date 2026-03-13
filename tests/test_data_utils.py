from pathlib import Path
import numpy as np
import pytest
import torch
from utils.data_utils import _read_subject, load_data, EEGDataset

DATA_DIR = Path("data")
SUBJECTS = [DATA_DIR / f"A0{i}T.gdf" for i in range(1, 10)]


class TestReadSubject:

    @pytest.fixture(scope="class")
    def subject_data(self):
        return _read_subject(SUBJECTS[0])

    def test_X_ndim(self, subject_data):
        X, _ = subject_data
        assert X.ndim == 3

    def test_X_channel_count(self, subject_data):
        X, _ = subject_data
        assert X.shape[1] == 22

    def test_X_sample_count(self, subject_data):
        X, _ = subject_data
        assert X.shape[2] == 1001

    def test_y_length_matches_X(self, subject_data):
        X, y = subject_data
        assert len(y) == X.shape[0]

    def test_labels_are_zero_indexed(self, subject_data):
        _, y = subject_data
        assert set(y).issubset({0, 1, 2, 3})

    def test_four_classes_present(self, subject_data):
        _, y = subject_data
        assert len(np.unique(y)) == 4

    def test_X_is_float(self, subject_data):
        X, _ = subject_data
        assert np.issubdtype(X.dtype, np.floating)

    def test_y_is_integer(self, subject_data):
        _, y = subject_data
        assert np.issubdtype(y.dtype, np.integer)


class TestLoadData:

    @pytest.fixture(scope="class")
    def all_data(self):
        return load_data(DATA_DIR)

    def test_X_is_float32(self, all_data):
        X, _ = all_data
        assert X.dtype == np.float32

    def test_all_subjects_loaded(self, all_data):
        X_single, _ = _read_subject(SUBJECTS[0])
        X, _ = all_data
        assert X.shape[0] > X_single.shape[0]


class TestEEGDataset:

    @pytest.fixture(scope="class")
    def dataset(self):
        X, y = load_data(DATA_DIR)
        return EEGDataset(X, y)

    def test_length_matches_loaded_data(self, dataset):
        X, _ = load_data(DATA_DIR)
        assert len(dataset) == X.shape[0]

    def test_item_shapes(self, dataset):
        x, y = dataset[0]
        assert x.shape == (22, 1001)
        assert y.shape == ()

    def test_X_is_float32_tensor(self, dataset):
        x, _ = dataset[0]
        assert x.dtype == torch.float32

    def test_y_is_long_tensor(self, dataset):
        _, y = dataset[0]
        assert y.dtype == torch.long

    def test_normalisation_mean_near_zero(self, dataset):
        x, _ = dataset[0]
        assert x.mean(dim=-1).abs().max() < 1e-5

    def test_normalisation_std_near_one(self, dataset):
        x, _ = dataset[0]
        assert (x.std(dim=-1) - 1).abs().max() < 1e-2