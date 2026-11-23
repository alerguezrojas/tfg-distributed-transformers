"""paravit doctor: evaluación pura sobre entornos simulados y comando de la CLI."""
import dataclasses

from typer.testing import CliRunner

from src import doctor as dr
from src.cli import app


def _env(**kw):
    base = dict(python=(3, 12, 3), torch_version="2.5.0", torch_cuda="11.8", cuda_available=True,
                driver="535.0", gpus=[dr.GpuInfo("Tesla V100-PCIE-32GB", 32.0)], avx2=True,
                machine="verode (--machine)")
    base.update(kw)
    return dr.Environment(**base)


def _by_name(checks):
    return {c.name: c for c in checks}


def test_healthy_gpu_machine_is_all_ok():
    checks = dr.evaluate(_env())
    assert {c.status for c in checks} == {dr.OK}
    assert dr.exit_code(checks) == 0


def test_no_avx2_fails_and_skips_torch():
    checks = _by_name(dr.evaluate(_env(avx2=False, torch_version=None, gpus=[],
                                       cuda_available=False)))
    assert checks["CPU AVX2"].status == dr.FAIL and "srun" in checks["CPU AVX2"].detail
    assert checks["torch"].status == dr.WARN          # not tried, not a second failure


def test_old_python_fails():
    assert _by_name(dr.evaluate(_env(python=(3, 10, 0))))["Python"].status == dr.FAIL


def test_cpu_only_machine_warns_unless_gpu_is_required():
    env = _env(gpus=[], cuda_available=False, torch_cuda=None)
    cuda = _by_name(dr.evaluate(env))["CUDA"]
    assert cuda.status == dr.WARN
    assert dr.exit_code(dr.evaluate(env)) == 0
    assert _by_name(dr.evaluate(env, require_gpu=True))["CUDA"].status == dr.FAIL
    assert dr.exit_code(dr.evaluate(env, require_gpu=True)) == 1
    # a config that needs NCCL requires a GPU on its own
    env = dataclasses.replace(env, config="c.yaml", wants_gpu=True)
    assert _by_name(dr.evaluate(env))["CUDA"].status == dr.FAIL


def test_gpu_with_cpu_build_of_torch_is_reported():
    env = _env(cuda_available=False, torch_cuda=None)
    cuda = _by_name(dr.evaluate(env))["CUDA"]
    assert cuda.status == dr.WARN and "CPU build" in cuda.detail


def test_gpu_present_but_unusable_by_torch():
    env = _env(cuda_available=False)                  # CUDA build, still unavailable
    cuda = _by_name(dr.evaluate(env))["CUDA"]
    assert cuda.status == dr.WARN and "mismatch" in cuda.detail


def test_unknown_gpu_warns():
    env = _env(gpus=[dr.GpuInfo("NVIDIA Imaginary 9000", 24.0)])
    assert _by_name(dr.evaluate(env))["GPU 0"].status == dr.WARN


def test_known_gpus_in_the_predictor_table():
    for name in ("NVIDIA GeForce RTX 3060 Ti", "Tesla T4", "Tesla V100-PCIE-32GB"):
        env = _env(gpus=[dr.GpuInfo(name)])
        assert _by_name(dr.evaluate(env))["GPU 0"].status == dr.OK, name


def test_no_profile_warns():
    assert _by_name(dr.evaluate(_env(machine=None)))["Machine profile"].status == dr.WARN


def test_config_paths_and_disk():
    env = _env(config="c.yaml", paths=[("data.root", "/d", True), ("data.metadata", "/m", False)],
               checkpoint_dir="checkpoints/x", free_gb=100.0, env_label="verode")
    checks = _by_name(dr.evaluate(env))
    assert checks["Path data.root"].status == dr.OK
    assert checks["Path data.metadata"].status == dr.FAIL
    assert checks["Checkpoint disk"].status == dr.OK
    assert dr.exit_code(dr.evaluate(env)) == 1
    for free, status in ((10.0, dr.WARN), (1.0, dr.FAIL), (None, dr.WARN)):
        env = dataclasses.replace(env, paths=[], free_gb=free)
        assert _by_name(dr.evaluate(env))["Checkpoint disk"].status == status


def test_without_config_no_config_checks():
    names = _by_name(dr.evaluate(_env()))
    assert not any(n.startswith("Path") or n == "Checkpoint disk" for n in names)


def test_config_wants_gpu():
    assert dr.config_wants_gpu({"distributed": {"backend": "nccl"}})
    assert not dr.config_wants_gpu({"distributed": {"backend": "gloo"}})
    assert not dr.config_wants_gpu({})


def test_collect_config_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(dr, "_nvidia_smi", lambda: ([], None))
    monkeypatch.setattr(dr, "_has_avx2", lambda: False)      # torch is not even imported
    root = tmp_path / "data"
    root.mkdir()
    cfg = {"data": {"root": str(root), "metadata": str(tmp_path / "no.parquet")},
           "checkpoint": {"dir": str(tmp_path / "ck")}, "output": {"env": "e"}}
    env = dr.collect(cfg, "c.yaml", "m (x)")
    assert env.torch_version is None and env.machine == "m (x)"
    assert env.paths == [("data.root", str(root), True),
                         ("data.metadata", str(tmp_path / "no.parquet"), False)]
    assert env.free_gb and env.env_label == "e"


def test_cli_doctor_exit_codes(tmp_path, monkeypatch):
    cfg = tmp_path / "c.yaml"
    cfg.write_text("data:\n  root: /no/existe/nunca\ncheckpoint:\n  dir: " + str(tmp_path) + "\n")
    import torch
    monkeypatch.setattr(dr, "_nvidia_smi", lambda: ([], None))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)    # simulated CPU-only machine
    monkeypatch.setattr(dr, "_has_avx2", lambda: True)
    res = CliRunner().invoke(app, ["doctor", "--config", str(cfg)])
    assert res.exit_code == 1 and "does not exist" in res.output
    res = CliRunner().invoke(app, ["doctor"])
    assert res.exit_code == 0 and "paravit doctor" in res.output
    res = CliRunner().invoke(app, ["doctor", "--require-gpu"])
    assert res.exit_code == 1
