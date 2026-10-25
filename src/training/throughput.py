"""Per-batch throughput — how long a training loop waits for data vs. computes.

:class:`StepTimer` splits the wall-clock time of each training iteration in two:

* ``data_time_s`` — waiting for the DataLoader: from the end of the previous
  iteration until the next batch is in hand.
* ``step_time_s`` — the rest of the step (transfer, forward, backward, optimizer and
  the metric reads that sync the GPU).

It only calls ``time.perf_counter()`` and never synchronises CUDA: the loops already
read ``loss.item()``/``.cpu()`` inside the step, so the host clock is the real time.
Batch hooks (event stream, CSV) run after ``step_done`` and are excluded from both
figures. Every loop that owns a training epoch (``Trainer``, the heterogeneous
trainer and the deep-tracing loop) uses this one helper.

In DDP the figures are per rank (the event stream is written by rank 0): the global
throughput is about ``samples_per_s × world_size``.
"""
import time


class StepTimer:
    """Usage inside a loop::

        timer = StepTimer()
        timer.start()
        for images, labels in loader:
            timer.batch_ready()
            ...step...
            batch_timing = timer.step_done(labels.shape[0])   # data_time_s, step_time_s, samples_per_s
            ...hooks...
            timer.iteration_end()
        result.update(timer.epoch_stats())                    # samples_per_s, data_wait_frac
    """

    def __init__(self):
        self.data_s = 0.0
        self.step_s = 0.0
        self.samples = 0
        self._mark = time.perf_counter()
        self._ready = self._mark
        self._data_batch = 0.0

    def start(self) -> None:
        """Before iterating the loader (the first wait starts here)."""
        self._mark = time.perf_counter()

    def batch_ready(self) -> None:
        """The loader has handed over the next batch."""
        self._ready = time.perf_counter()
        self._data_batch = self._ready - self._mark

    def step_done(self, n_samples: int) -> dict:
        """The step finished (after the GPU sync); returns this batch's figures."""
        step = time.perf_counter() - self._ready
        self.data_s += self._data_batch
        self.step_s += step
        self.samples += int(n_samples)
        total = self._data_batch + step
        return {
            "data_time_s": self._data_batch,
            "step_time_s": step,
            "samples_per_s": n_samples / total if total > 0 else 0.0,
        }

    def iteration_end(self) -> None:
        """After the batch hooks: the next data wait is counted from here."""
        self._mark = time.perf_counter()

    def epoch_stats(self) -> dict:
        """``samples_per_s`` and ``data_wait_frac`` (share of the time waiting for data)."""
        total = self.data_s + self.step_s
        return {
            "samples_per_s": self.samples / total if total > 0 else 0.0,
            "data_wait_frac": self.data_s / total if total > 0 else 0.0,
        }
