import functools
import time


def timed(fn):
    """Log the wall-clock execution time of any function.

    Adds 'time' key to the result dict if it is a dict.

    Example:
        trainer.train_epoch = timed(trainer.train_epoch)
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[timed] {fn.__qualname__}: {elapsed:.2f}s")
        if isinstance(result, dict) and "time" not in result:
            result = {**result, "time": elapsed}
        return result
    return wrapper
