"""Instantánea de los comandos de ``paravit train`` (lo que muestra ``--dry-run``) por estrategia.

Fija el argv exacto de cada estrategia para que el refactor del lanzador (registro de estrategias)
no cambie lo que ejecutan la CLI, el menú y los comandos que el autor ya tiene escritos.
"""
import sys

import pytest

from src.cli import build_train_cmd

PY = sys.executable
FULL = dict(model="vit_tiny_patch16_224", epochs=3, trace="off", precision="amp",
            layers=["plot", "confusion"], fn=["timing"], monitor=True, seed=1, data_root="/d",
            machine="m", overrides=["training.lr=0.1"], n_gpus=2, nnodes=2, node_rank=1,
            master_addr="h", master_port=1234)
_COMMON = ["--config", "c.yaml", "--model", "vit_tiny_patch16_224", "--epochs", "3", "--seed", "1",
           "--data-root", "/d", "--machine", "m", "--set", "training.lr=0.1", "--trace", "off",
           "--monitor"]
_LAYERS_FN = ["--layers", "plot", "confusion", "--fn", "timing"]
_RUN = [PY, "-m", "torch.distributed.run"]
_RDZV = ["--nnodes=2", "--node_rank=1", "--master_addr=h", "--master_port=1234"]

FULL_CMDS = {
    "single": [PY, "scripts/train_single_gpu.py", *_COMMON, "--precision", "amp", *_LAYERS_FN],
    "ddp": [*_RUN, "--nproc_per_node=2", *_RDZV, "scripts/train_ddp.py", *_COMMON,
            "--precision", "amp", *_LAYERS_FN],
    "model-parallel": [PY, "scripts/train_model_parallel.py", *_COMMON, "--precision", "amp",
                       *_LAYERS_FN],
    # un proceso por nodo y sin --precision (su bucle es fp32)
    "heterogeneous": [*_RUN, "--nproc_per_node=1", *_RDZV, "scripts/train_heterogeneous_ddp.py",
                      *_COMMON, *_LAYERS_FN],
}
MIN_CMDS = {
    "single": [PY, "scripts/train_single_gpu.py", "--config", "c.yaml", "--trace", "simple"],
    "ddp": [*_RUN, "--nproc_per_node=2", "scripts/train_ddp.py", "--config", "c.yaml",
            "--trace", "simple"],
    "model-parallel": [PY, "scripts/train_model_parallel.py", "--config", "c.yaml",
                       "--trace", "simple"],
    "heterogeneous": [*_RUN, "--nproc_per_node=1", "scripts/train_heterogeneous_ddp.py",
                      "--config", "c.yaml", "--trace", "simple"],
}


@pytest.mark.parametrize("strategy", sorted(FULL_CMDS))
def test_train_command_with_every_option(strategy):
    assert build_train_cmd(strategy, "c.yaml", **FULL) == FULL_CMDS[strategy]


@pytest.mark.parametrize("strategy", sorted(MIN_CMDS))
def test_train_command_defaults(strategy):
    assert build_train_cmd(strategy, "c.yaml") == MIN_CMDS[strategy]
