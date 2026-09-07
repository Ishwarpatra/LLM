"""Shared pytest fixtures for all HYDRA-LM tests."""
import pytest
import torch
from hydra_lm.config import HydraConfig


@pytest.fixture
def cfg():
    """Toy HydraConfig — runs on CPU in well under 1 second."""
    return HydraConfig.toy()


@pytest.fixture
def B():
    return 4

@pytest.fixture
def T():
    return 20

@pytest.fixture
def device():
    return torch.device("cpu")
