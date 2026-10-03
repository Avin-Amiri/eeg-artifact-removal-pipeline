import numpy as np
import pytest
import torch

from src.features import extract_band_powers
from src.model import EEGNetLight


def test_eegnet_forward_dimensions():
    batch_size = 8
    n_channels = 32
    n_samples = 256
    num_classes = 2

    model = EEGNetLight(
        num_channels=n_channels, num_classes=num_classes, samples=n_samples
    )
    dummy_input = torch.randn(batch_size, 1, n_channels, n_samples)

    out = model(dummy_input)

    assert out.shape == (
        batch_size,
        num_classes,
    ), f"Expected shape {(batch_size, num_classes)}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN values"


def test_model_different_samples():
    samples = 128
    model = EEGNetLight(num_channels=16, num_classes=3, samples=samples)
    dummy_input = torch.randn(4, 1, 16, samples)
    out = model(dummy_input)
    assert out.shape == (4, 3)
