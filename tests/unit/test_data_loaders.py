"""
Tests for simple CIFAR-like data loading utilities.
"""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset


class DummyCIFARDataset(Dataset):
    def __init__(self, n: int = 32):
        self.n = n
        self.data = torch.randn(n, 3, 32, 32)
        self.labels = torch.randint(0, 10, (n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.n:
            raise IndexError("index out of range")
        return self.data[idx], self.labels[idx]


class TestDatasetBasics:
    def test_len_and_getitem(self):
        ds = DummyCIFARDataset(n=10)
        assert len(ds) == 10
        x, y = ds[0]
        assert x.shape == (3, 32, 32)
        assert 0 <= y.item() < 10

    def test_out_of_bounds(self):
        ds = DummyCIFARDataset(n=5)
        with pytest.raises(IndexError):
            _ = ds[5]


class TestDataLoaderBasics:
    def test_dataloader_batches(self):
        ds = DummyCIFARDataset(n=20)
        loader = DataLoader(ds, batch_size=6, shuffle=False)
        x, y = next(iter(loader))
        assert x.shape[0] <= 6
        assert x.shape[1:] == (3, 32, 32)
        assert y.shape[0] == x.shape[0]
