"""Python @ function decorators for cross-cutting concerns.

These are standard Python function decorators (using the @ syntax),
as opposed to the OOP Decorator pattern in trainer_decorators.py.
Both solve the same problem — adding behaviour without modifying the
original code — but at different levels of abstraction.

  OOP Decorator pattern → wraps entire objects / classes
  Python @ decorators   → wraps individual functions / methods
"""

import functools
import time


def timed(fn):
    """Log the execution time of any function.

    Example:
        @timed
        def train_epoch(self, loader): ...
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[timed] {fn.__qualname__} → {elapsed:.2f}s")
        return result
    return wrapper


def log_call(fn):
    """Log when a function is called and when it returns.

    Example:
        @log_call
        def eval_epoch(self, loader): ...
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[call] → {fn.__qualname__}")
        result = fn(*args, **kwargs)
        print(f"[call] ← {fn.__qualname__} done")
        return result
    return wrapper


def retry_on_cuda_oom(fn):
    """Retry with halved batch size on CUDA out-of-memory error.

    Useful during distributed training where memory pressure is higher.

    Example:
        @retry_on_cuda_oom
        def train_epoch(self, loader): ...
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        import torch
        try:
            return fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[oom] CUDA OOM en {fn.__qualname__} — liberando caché y reintentando")
            return fn(*args, **kwargs)
    return wrapper
