from __future__ import annotations

import numpy as np
import pytest
import torch

from scripts.check_gpu_environment import check_environment
from scripts.run_cet_benchmark import environment
from src.training.engine import resolve_device, set_seed


def test_cpu_device_resolution_and_environment_metadata():
    device=resolve_device("cpu")
    assert str(device)=="cpu"
    metadata=environment(device)
    assert metadata["device"]=="cpu"
    assert "deterministic_algorithms" in metadata
    assert "torch_cuda" in metadata


def test_requesting_unavailable_cuda_fails_explicitly():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this test environment")
    with pytest.raises(RuntimeError,match="CUDA device"):
        resolve_device("cuda:0")


def test_seeded_deterministic_cpu_replay_is_exact():
    set_seed(123,deterministic=True,warn_only=False)
    first=np.random.normal(size=5)
    first_torch=torch.randn(5)
    set_seed(123,deterministic=True,warn_only=False)
    assert np.array_equal(first,np.random.normal(size=5))
    assert torch.equal(first_torch,torch.randn(5))


def test_gpu_preflight_can_be_exercised_on_cpu_without_claiming_eligibility():
    result=check_environment("cpu",allow_cpu=True)
    assert result["deterministic_training_probe"]["exact_replay"] is True
    assert result["eligible_for_publication_gpu_run"] is False
