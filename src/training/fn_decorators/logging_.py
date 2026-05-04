import functools


def log_call(fn):
    """Log when a function is called and when it returns.

    Example:
        trainer.eval_epoch = log_call(trainer.eval_epoch)
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[call] → {fn.__qualname__}")
        result = fn(*args, **kwargs)
        print(f"[call] ← {fn.__qualname__} done")
        return result
    return wrapper
