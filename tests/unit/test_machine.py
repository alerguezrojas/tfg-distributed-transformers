"""Perfiles de máquina: selección, aplicación por dataset, precedencia y --set."""
import argparse
import copy
import re
from pathlib import Path

import pytest
import yaml

from src.machine import (MachineProfile, apply_profile, available_profiles,
                         describe, load_profile, parse_profile, select_machine)
from src.training.config_overrides import (apply_set_overrides, parse_set_value, prepare_config)

ROOT = Path(__file__).resolve().parents[2]


def _write(d: Path, name: str, **body) -> None:
    (d / f"{name}.yaml").write_text(yaml.safe_dump(body))


@pytest.fixture
def machines(tmp_path):
    d = tmp_path / "machines"
    d.mkdir()
    _write(d, "alpha", env="alpha", hosts=["alpha-*"], checkpoint_dir="/ck/alpha", num_workers=3,
           data_roots={"bigearthnet": {"root": "/a/ben", "metadata_dir": "/a/meta"},
                       "panda": "/a/panda", "medmnist": "/a/med"})
    _write(d, "beta", env="beta", hosts=["beta*", "*.cluster"])
    return d


def _ben_cfg():
    return {"data": {"root": "/orig", "metadata": "/orig/metadata_demo.parquet", "num_workers": 8},
            "checkpoint": {"dir": "checkpoints/local", "keep": "best"}, "output": {"env": "local"},
            "training": {"lr": 0.1}}


# ── selección ────────────────────────────────────────────────────────────────

def test_selection_cli_beats_env_beats_host(machines):
    env = {"PARAVIT_MACHINE": "beta"}
    sel = select_machine("alpha", env, "beta1", machines)
    assert (sel.profile.name, sel.reason) == ("alpha", "--machine")
    sel = select_machine(None, env, "alpha-1", machines)
    assert (sel.profile.name, sel.reason) == ("beta", "PARAVIT_MACHINE")
    sel = select_machine(None, {}, "alpha-1", machines)
    assert sel.profile.name == "alpha" and sel.reason.startswith("host")


def test_selection_host_patterns_case_insensitive(machines):
    assert select_machine(None, {}, "BETA7", machines).profile.name == "beta"
    assert select_machine(None, {}, "node.cluster", machines).profile.name == "beta"


def test_selection_none_when_nothing_matches(machines):
    assert select_machine(None, {}, "otro-pc", machines) is None


def test_none_disables_even_host_match(machines):
    assert select_machine("none", {}, "alpha-1", machines) is None
    assert select_machine(None, {"PARAVIT_MACHINE": "none"}, "alpha-1", machines) is None


def test_unknown_profile_lists_available(machines):
    with pytest.raises(SystemExit, match="alpha, beta"):
        select_machine("gamma", {}, "x", machines)


def test_profile_name_is_not_a_path(machines):
    with pytest.raises(SystemExit, match="no válido"):
        load_profile("../machines/alpha", machines)


def test_invalid_profile_contents():
    with pytest.raises(SystemExit, match="desconocidas"):
        parse_profile("x", {"checkpoint_dirr": "a"})
    with pytest.raises(SystemExit, match="num_workers"):
        parse_profile("x", {"num_workers": -1})
    with pytest.raises(SystemExit, match="hosts"):
        parse_profile("x", {"hosts": "verode*"})


def test_bundled_profiles_are_valid():
    names = available_profiles()
    assert {"local", "kaggle", "verode", "example"} <= set(names)
    for n in names:
        load_profile(n)
    assert load_profile("verode").env == "verode"


def test_bundled_hosts_select_the_right_profile():
    assert select_machine(None, {}, "pc-ubuntu").profile.name == "local"
    assert select_machine(None, {}, "verode00").profile.name == "verode"
    assert select_machine(None, {}, "un-pc-cualquiera") is None


# ── aplicación ───────────────────────────────────────────────────────────────

def test_apply_bigearthnet_keeps_metadata_file_name(machines):
    cfg = _ben_cfg()
    changes = apply_profile(cfg, load_profile("alpha", machines))
    assert cfg["data"]["root"] == "/a/ben"
    assert cfg["data"]["metadata"] == "/a/meta/metadata_demo.parquet"   # demo stays demo
    assert cfg["data"]["num_workers"] == 8   # the experiment value stays
    assert cfg["checkpoint"] == {"dir": "/ck/alpha", "keep": "best"}
    assert cfg["output"]["env"] == "alpha"
    assert any(c.startswith("data.root:") for c in changes)


@pytest.mark.parametrize("name, expected", [("panda", "/a/panda"), ("medmnist_chest", "/a/med"),
                                            ("medmnist_derma", "/a/med"),
                                            ("bigearthnet", "/a/ben")])
def test_apply_by_dataset_name(machines, name, expected):
    cfg = {"data": {"name": name, "root": "/orig"}}
    apply_profile(cfg, load_profile("alpha", machines))
    assert cfg["data"]["root"] == expected


def test_default_dataset_is_bigearthnet(machines):
    cfg = _ben_cfg()                                    # no data.name
    apply_profile(cfg, load_profile("alpha", machines))
    assert cfg["data"]["root"] == "/a/ben"


def test_dataset_not_in_profile_keeps_its_paths(machines):
    cfg = {"data": {"name": "otro", "root": "/orig"}, "checkpoint": {"dir": "c"}}
    apply_profile(cfg, load_profile("alpha", machines))
    assert cfg["data"]["root"] == "/orig"
    assert cfg["checkpoint"]["dir"] == "/ck/alpha"      # machine-wide settings still apply


def test_metadata_dir_without_metadata_in_config_is_skipped(machines):
    cfg = {"data": {"root": "/orig"}}
    apply_profile(cfg, load_profile("alpha", machines))
    assert "metadata" not in cfg["data"]


def test_apply_is_idempotent(machines):
    cfg = _ben_cfg()
    prof = load_profile("alpha", machines)
    apply_profile(cfg, prof)
    assert apply_profile(cfg, prof) == []


def test_empty_profile_changes_nothing():
    cfg = _ben_cfg()
    before = copy.deepcopy(cfg)
    assert apply_profile(cfg, MachineProfile(name="vacio")) == []
    assert cfg == before


def test_describe():
    assert "none" in describe(None, [])
    sel = select_machine("local", {})
    assert "local (--machine)" in describe(sel, ["data.root: 'a' -> 'b'"])


def test_broken_profile_does_not_break_host_search(machines, capsys):
    (machines / "aaa_roto.yaml").write_text("checkpoint_dirr: x\nhosts: ['zzz*']\n")
    assert select_machine(None, {}, "alpha-1", machines).profile.name == "alpha"
    assert "aaa_roto" in capsys.readouterr().err
    with pytest.raises(SystemExit, match="desconocidas"):      # explicit request: full error
        select_machine("aaa_roto", {}, "x", machines)


def test_config_of_this_machine_is_untouched(machines):
    cfg = _ben_cfg()
    cfg["output"]["env"] = "alpha"
    before = copy.deepcopy(cfg)
    assert apply_profile(cfg, load_profile("alpha", machines)) == []
    assert cfg == before


def test_num_workers_only_fills_a_gap(machines):
    cfg = {"data": {"name": "panda", "root": "/o", "num_workers": 8}, "output": {"env": "x"}}
    apply_profile(cfg, load_profile("alpha", machines))
    assert cfg["data"]["num_workers"] == 8
    cfg = {"data": {"name": "panda", "root": "/o"}, "output": {"env": "x"}}
    apply_profile(cfg, load_profile("alpha", machines))
    assert cfg["data"]["num_workers"] == 3


ALL_CONFIGS = sorted((ROOT / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.name)
def test_every_config_is_stable_under_its_own_machine_and_keeps_workers(path):
    cfg = yaml.safe_load(path.read_text())
    for name in available_profiles():
        prof = load_profile(name)
        trial = copy.deepcopy(cfg)
        changes = apply_profile(trial, prof)
        if (cfg.get("output") or {}).get("env") == prof.env:
            assert changes == [] and trial == cfg, (path.name, name)
        if "num_workers" in cfg.get("data", {}):
            assert trial["data"]["num_workers"] == cfg["data"]["num_workers"], (path.name, name)


def test_bundled_local_matches_the_paths_in_existing_configs():
    """El perfil local reproduce las rutas que ya traen las configs del autor."""
    prof = load_profile("local")
    for name in ("train_v3.yaml", "train_panda.yaml", "medmnist_chest_local.yaml"):
        cfg = yaml.safe_load((ROOT / "configs" / name).read_text())
        before = copy.deepcopy(cfg)
        apply_profile(cfg, prof)
        for key in ("root", "metadata"):
            assert cfg["data"].get(key) == before["data"].get(key), (name, key)
        assert cfg["checkpoint"]["dir"] == before["checkpoint"]["dir"]
        assert cfg["output"]["env"] == before["output"]["env"]


def test_bundled_verode_matches_the_cluster_configs():
    prof = load_profile("verode")
    cfg = yaml.safe_load((ROOT / "configs" / "train_cluster_v3.yaml").read_text())
    before = copy.deepcopy(cfg)
    apply_profile(cfg, prof)
    assert cfg["data"]["root"] == before["data"]["root"]
    assert cfg["data"]["metadata"] == before["data"]["metadata"]


# ── --set ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, expected", [
    ("1e-4", 1e-4), ("1E-4", 1e-4), ("5e3", 5000.0), ("0.0001", 0.0001), ("1.0e-4", 1e-4),
    ("32", 32), ("true", True), ("false", False), ("null", None), ("[1, 2]", [1, 2]),
    ("cuda:0", "cuda:0"), ("abc", "abc"), ("1e", "1e"),
    ("[1e-4, 2]", [1e-4, 2]), ("{lr: 1e-3, n: 3}", {"lr": 1e-3, "n": 3}),
    ('"1e-4"', "1e-4"), ("'1e-4'", "1e-4"), ('["1e-4", 1e-4]', ["1e-4", 1e-4]),
])
def test_parse_set_value(text, expected):
    got = parse_set_value(text)
    assert got == expected and type(got) is type(expected)
    if isinstance(expected, list):
        assert [type(x) for x in got] == [type(x) for x in expected]


def test_set_overrides_nested_and_creates_sections():
    cfg = {"training": {"lr": 0.1}}
    lines = apply_set_overrides(cfg, ["training.lr=1e-4", "data.num_workers=2",
                                      "model.opts.flags=[a, b]"])
    assert cfg == {"training": {"lr": 1e-4}, "data": {"num_workers": 2},
                   "model": {"opts": {"flags": ["a", "b"]}}}
    assert "new key" not in lines[0] and "new key" in lines[1]


def test_set_value_may_contain_equals():
    cfg = {}
    apply_set_overrides(cfg, ["a.b=x=y"])
    assert cfg == {"a": {"b": "x=y"}}


@pytest.mark.parametrize("bad", ["lr", "=3", "a..b=1", "a.=1"])
def test_set_rejects_malformed(bad):
    with pytest.raises(SystemExit, match="clave.sub=valor"):
        apply_set_overrides({}, [bad])


def test_set_through_non_section_fails():
    with pytest.raises(SystemExit, match="no es una sección"):
        apply_set_overrides({"training": {"lr": 0.1}}, ["training.lr.x=1"])


# ── prepare_config: el único sitio de todos los puntos de entrada ────────────────

def _args(path, **kw):
    ns = argparse.Namespace(config=str(path), data_root=None, seed=None, epochs=None,
                            machine=None, set_=[])
    ns.__dict__.update(kw)
    return ns


@pytest.fixture
def cfg_file(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(_ben_cfg() | {"model": {"name": "vit_tiny_patch16_224"}}))
    return p


def test_no_profile_leaves_config_intact(cfg_file, capsys):
    cfg = prepare_config(_args(cfg_file))               # conftest: PARAVIT_MACHINE=none
    assert cfg == yaml.safe_load(cfg_file.read_text())
    assert "none (config used as written)" in capsys.readouterr().out


def test_profile_then_explicit_option_then_set(cfg_file, monkeypatch, capsys):
    mdir = cfg_file.parent / "m"
    mdir.mkdir()
    monkeypatch.setattr("src.machine.MACHINES_DIR", mdir)
    _write(mdir, "p", env="p", data_roots={"bigearthnet": "/prof"}, num_workers=2,
           checkpoint_dir="/ck")
    monkeypatch.setenv("PARAVIT_MACHINE", "p")
    cfg = prepare_config(_args(cfg_file))
    assert (cfg["data"]["root"], cfg["data"]["num_workers"], cfg["output"]["env"]) == ("/prof", 8, "p")
    assert "Machine profile: p (PARAVIT_MACHINE)" in capsys.readouterr().out
    # the explicit option beats the profile; --set goes last
    cfg = prepare_config(_args(cfg_file, data_root="/cli", set_=["data.num_workers=6"]))
    assert cfg["data"]["root"] == "/cli" and cfg["data"]["num_workers"] == 6
    # --machine beats the variable
    _write(mdir, "q", env="q")
    assert prepare_config(_args(cfg_file, machine="q"))["output"]["env"] == "q"


def test_set_scientific_float_reaches_config(cfg_file):
    cfg = prepare_config(_args(cfg_file, set_=["training.lr=1e-4"]))
    assert cfg["training"]["lr"] == 1e-4 and isinstance(cfg["training"]["lr"], float)


def test_only_rank_zero_prints(cfg_file, monkeypatch, capsys):
    monkeypatch.setenv("RANK", "1")
    prepare_config(_args(cfg_file))
    assert capsys.readouterr().out == ""


_TRAIN_SCRIPTS = ["train_single_gpu", "train_ddp", "train_heterogeneous_ddp", "train_model_parallel"]


@pytest.mark.parametrize("script", ["eval", "predict", "benchmark"])
def test_every_entry_point_goes_through_prepare_config(script):
    src = (ROOT / "scripts" / f"{script}.py").read_text()
    assert "prepare_config(" in src
    assert re.search(r"add_machine_options|add_common_overrides", src)
    assert "safe_load" not in src                       # no YAML read bypassing the profile


def test_training_launcher_goes_through_prepare_config():
    """The four training scripts are wrappers of src/training/launch.py: that is where the config
    is built (profile + overrides) and the options registered."""
    src = (ROOT / "src" / "training" / "launch.py").read_text()
    assert "prepare_config(" in src and "add_common_overrides" in src
    assert "safe_load" not in src
    for script in _TRAIN_SCRIPTS:
        wrapper = (ROOT / "scripts" / f"{script}.py").read_text()
        assert "from src.training.launch import" in wrapper and "safe_load" not in wrapper
