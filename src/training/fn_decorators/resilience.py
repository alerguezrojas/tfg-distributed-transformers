import functools


def retry_on_cuda_oom(fn):
    """Retry the function once after clearing the CUDA cache on OOM.

    Useful during distributed training where memory pressure is higher.

    Example:
        trainer.train_epoch = retry_on_cuda_oom(trainer.train_epoch)
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
