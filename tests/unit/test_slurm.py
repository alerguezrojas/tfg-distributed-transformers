"""Lanzador Slurm: lista de nodos, comando torchrun c10d y errores (sin Slurm, solo CPU)."""
import sys

import pytest

from src.cli import build_train_cmd
from src.slurm import expand_nodelist, first_node, in_slurm_job, slurm_torchrun

ENV = {"SLURM_JOB_ID": "777", "SLURM_JOB_NODELIST": "verode[16,21]", "SLURM_NNODES": "2"}
_RUN = [sys.executable, "-m", "torch.distributed.run"]


@pytest.mark.parametrize("expr, expected", [
    ("verode16", ["verode16"]),
    ("verode[16,21]", ["verode16", "verode21"]),
    ("n[01-03],m5", ["n01", "n02", "n03", "m5"]),
    ("n[1,3-4]x", ["n1x", "n3x", "n4x"]),
    ("a[08-10]", ["a08", "a09", "a10"]),
    ("a,b[1-2],c", ["a", "b1", "b2", "c"]),
])
def test_expand_nodelist(expr, expected):
    assert expand_nodelist(expr) == expected


@pytest.mark.parametrize("bad", ["", "n[1-", "n[a-b]", "n[3-1]", "n[1]]"])
def test_expand_nodelist_invalid(bad):
    with pytest.raises(ValueError):
        expand_nodelist(bad)


def test_first_node_without_scontrol():
    assert first_node(ENV, use_scontrol=False) == "verode16"
    with pytest.raises(ValueError, match="SLURM_JOB_NODELIST"):
        first_node({}, use_scontrol=False)


def test_in_slurm_job():
    assert in_slurm_job(ENV)
    assert not in_slurm_job({"SLURM_JOB_ID": "1"})


def test_slurm_torchrun_is_same_on_every_node():
    a = slurm_torchrun(1, {**ENV, "SLURMD_NODENAME": "verode16"}, use_scontrol=False)
    b = slurm_torchrun(1, {**ENV, "SLURMD_NODENAME": "verode21"}, use_scontrol=False)
    assert a == b == [*_RUN, "--nnodes=2", "--nproc_per_node=1", "--rdzv-backend=c10d",
                      "--rdzv-endpoint=verode16:29500", "--rdzv-id=777"]


def test_slurm_outside_job_fails():
    with pytest.raises(ValueError, match="dentro de un trabajo de Slurm"):
        slurm_torchrun(1, {})


def test_build_train_cmd_slurm(monkeypatch):
    monkeypatch.setattr("src.slurm.shutil.which", lambda _: None)   # parser propio
    cmd = build_train_cmd("ddp", "c.yaml", n_gpus=2, master_port=1234, launcher="slurm", env=ENV)
    assert cmd[:7] == [*_RUN, "--nnodes=2", "--nproc_per_node=2", "--rdzv-backend=c10d",
                       "--rdzv-endpoint=verode16:1234"]
    assert not any(a.startswith(("--node_rank", "--master_addr")) for a in cmd)
    assert cmd[cmd.index("scripts/train_ddp.py") + 1:][:2] == ["--config", "c.yaml"]


def test_build_train_cmd_slurm_errors():
    with pytest.raises(ValueError, match="torchrun"):
        build_train_cmd("single", "c.yaml", launcher="slurm", env=ENV)
    with pytest.raises(ValueError, match="Slurm"):
        build_train_cmd("ddp", "c.yaml", launcher="slurm", env={})
    with pytest.raises(ValueError, match="launcher"):
        build_train_cmd("ddp", "c.yaml", launcher="pbs")


def test_local_launcher_is_default_and_identical():
    """Sin --launcher (o con local) el comando es el de siempre, aunque haya variables Slurm."""
    kw = dict(n_gpus=2, nnodes=2, node_rank=1, master_addr="h", master_port=1234)
    base = build_train_cmd("ddp", "c.yaml", **kw)
    assert build_train_cmd("ddp", "c.yaml", launcher="local", env=ENV, **kw) == base
    assert base[:6] == [*_RUN, "--nproc_per_node=2", "--nnodes=2", "--node_rank=1"]
