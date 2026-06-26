"""retry_on_cuda_oom: recover once after clearing the CUDA cache, then re-raise."""
import pytest
import torch

from src.training.fn_decorators import retry_on_cuda_oom


def test_retry_on_cuda_oom_recovers_once(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)  # no GPU in CI
    calls = {"n": 0}

    @retry_on_cuda_oom
    def f():
        calls["n"] += 1
        if calls["n"] == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return "ok"

    assert f() == "ok"
    assert calls["n"] == 2          # original call + one retry


def test_retry_on_cuda_oom_reraises_after_one_retry(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    calls = {"n": 0}

    @retry_on_cuda_oom
    def f():
        calls["n"] += 1
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    with pytest.raises(torch.cuda.OutOfMemoryError):
        f()
    assert calls["n"] == 2          # retries exactly once, then re-raises
