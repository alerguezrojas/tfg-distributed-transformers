"""Helpers puros de la página Inspector (src/web/inspector_data.py) y tipos nuevos del lector."""
import numpy as np
import pytest

from src.web import inspector_data as d
from src.web.events_reader import EventsFollower, RunState
from tests.fixtures.make_inspector_run import make_inspector_run


def _sys(t, powers):
    return {"type": "system", "time": t, "cpu_pct": 10.0, "ram_used_pct": 20.0,
            "proc_cpu_pct": 100.0, "proc_rss_mb": 500.0,
            "gpus": [{"index": i, "name": "G", "util_pct": 50.0, "mem_used_mb": 1000.0,
                      "mem_total_mb": 4000.0, "power_w": p, "temp_c": 60.0, "sm_clock_mhz": 1500.0}
                     for i, p in enumerate(powers)]}


def test_system_frames_and_energy():
    evs = [_sys(100.0, [100.0, 50.0]), _sys(110.0, [100.0, 50.0]), _sys(130.0, [200.0, 50.0])]
    gpu, host = d.system_frames(evs)
    assert len(gpu) == 6 and len(host) == 3
    assert gpu["t_s"].max() == 30.0 and gpu["mem_pct"].iloc[0] == 25.0
    wh = d.energy_wh(gpu)
    assert wh[0] == pytest.approx((10 * 100 + 20 * 150) / 3600)
    assert wh[1] == pytest.approx(30 * 50 / 3600)
    summ = d.system_summary(gpu)
    assert summ["Power peak (W)"].tolist() == [200.0, 50.0]


def test_system_frames_without_gpus_and_empty():
    gpu, host = d.system_frames([{"type": "system", "time": 1.0, "gpus": [], "cpu_pct": 5}])
    assert gpu.empty and len(host) == 1
    assert d.energy_wh(gpu) == {} and d.system_summary(gpu).empty
    gpu, host = d.system_frames([])
    assert gpu.empty and host.empty


def test_diagnose_data_wait_thresholds():
    assert d.diagnose_data_wait(0.31)[0] == "bad" and "I/O-bound" in d.diagnose_data_wait(0.5)[1]
    assert d.diagnose_data_wait(0.3)[0] == "warn"          # el umbral es estricto (> 0,3)
    assert d.diagnose_data_wait(0.05)[0] == "ok"
    assert d.diagnose_data_wait(None)[0] == "none"
    assert d.diagnose_data_wait(float("nan"))[0] == "none"


def test_batch_and_epoch_performance():
    rows = [{"progress": 0.1, "samples_per_s": 100.0, "data_time_s": 0.1, "step_time_s": 0.3},
            {"progress": 0.2, "samples_per_s": 110.0, "data_time_s": 0.0, "step_time_s": 0.0}]
    df = d.batch_performance(rows)
    assert df["data_frac"].iloc[0] == pytest.approx(0.25) and np.isnan(df["data_frac"].iloc[1])
    assert d.batch_performance([{"progress": 1.0, "batch_loss": 0.3}]).empty
    ends = [{"epoch": 1, "train": {"samples_per_s": 200.0, "data_wait_frac": 0.2}},
            {"epoch": 2, "train": {"loss": 1}}]
    ep = d.epoch_performance(ends)
    assert ep["epoch"].tolist() == [1] and ep["data_wait_frac"].iloc[0] == 0.2


def test_layer_matrix_orders_and_thins():
    evs = [{"epoch": 1, "batch": 5, "groups": [{"name": "a", "grad_norm": 1.0},
                                               {"name": "b", "grad_norm": 2.0}]},
           {"epoch": 1, "batch": 10, "groups": [{"name": "a", "grad_norm": 3.0}]},
           {"epoch": 2, "batch": 10, "groups": [{"name": "b", "grad_norm": 4.0}]}]
    x, groups, z = d.layer_matrix(evs, "grad_norm", {1: 10, 2: 10})
    assert groups == ["a", "b"] and x.tolist() == [0.5, 1.0, 2.0]
    assert z[0].tolist()[:2] == [1.0, 3.0] and np.isnan(z[0, 2]) and np.isnan(z[1, 1])
    many = [{"epoch": 1, "batch": i, "groups": [{"name": "a", "grad_norm": 1.0}]}
            for i in range(1, 1001)]
    x, _, z = d.layer_matrix(many, "grad_norm", max_cols=100)
    assert len(x) <= 100 and z.shape[1] == len(x) and x[0] < x[-1]
    assert d.layer_matrix([], "grad_norm")[1] == []


def test_update_ratio_status_and_histograms():
    assert [d.update_ratio_status(v) for v in (1e-6, 1e-3, 0.5, 0.0, None)] == \
        ["low", "healthy", "high", "unknown", "unknown"]
    h = {2: {"groups": [{"name": "g", "edges": list(range(33)), "counts": [1] * 32},
                        {"name": "bad", "edges": [0, 1], "counts": [1, 2]}]}}
    out = d.histograms_for_epoch(h, 2)
    assert list(out) == ["g"] and d.histograms_for_epoch(h, 9) == {}


def test_prediction_table_filters_and_flips():
    probs = np.array([[.9, .1, 0], [.2, .7, .1], [.1, .2, .7], [.6, .3, .1]])
    targets = np.eye(3, dtype=np.uint8)[[0, 0, 2, 0]]
    prev = np.array([[.9, .1, 0], [.8, .1, .1], [.7, .2, .1], [.6, .3, .1]])
    t = d.prediction_table(probs, targets, [0.1, 0.9, 0.2, 0.5], "multiclass",
                           ["a", "b", "c"], prev_probs=prev)
    assert t["correct"].tolist() == [True, False, True, True]
    assert t["flip"].tolist() == ["", "to_wrong", "to_right", ""]
    assert t["pred"].tolist() == ["a", "b", "c", "a"]
    assert d.filter_table(t, outcome="wrong")["pos"].tolist() == [1]
    assert d.filter_table(t, true_class=0, pred_class=0)["pos"].tolist() == [0, 3]
    assert d.filter_table(t, flips_only="changed")["pos"].tolist() == [1, 2]
    assert d.filter_table(t, flips_only="to_right")["pos"].tolist() == [2]
    assert d.filter_table(t, sort="Loss (highest first)")["pos"].tolist() == [1, 3, 2, 0]


def test_prediction_table_binary_and_multilabel():
    b = d.prediction_table([[0.8], [0.3]], [[1], [1]], [0.2, 1.2], "binary", ["pos"])
    assert b["pred"].tolist() == ["positive", "negative"]
    assert b["correct"].tolist() == [True, False]
    assert b["confidence"].tolist() == pytest.approx([0.8, 0.7])
    m = d.prediction_table([[0.9, 0.1], [0.1, 0.1]], [[1, 0], [0, 0]], [0, 0], "multilabel",
                           ["x", "y"])
    assert m["pred"].tolist() == ["x", "none"] and m["true"].tolist() == ["x", "none"]


def test_trajectory_and_labels():
    tr = d.trajectory_frame([1, 2], [np.array([[.1, .9]]), np.array([[.4, .6]])], 0, ["a", "b"])
    assert len(tr) == 4 and tr[tr["epoch"] == 2]["prob"].tolist() == pytest.approx([0.4, 0.6])
    y = np.array([[1, 0], [1, 1], [0, 0]])
    assert d.primary_label(y, ["a", "b"]).tolist() == ["a", "multiple", "none"]
    assert d.predicted_label([[.2, .8]], ["a", "b"]).tolist() == ["b"]


def test_images():
    img = np.zeros((8, 8, 3), np.uint8)
    assert d.upscale(img, 32).shape == (32, 32, 3)
    card = d.outcome_card(img, True, 32)
    assert tuple(card[0, 0]) == d.GREEN and tuple(card[16, 16]) == (0, 0, 0)
    assert tuple(d.outcome_card(img, False, 32)[0, 0]) == d.RED
    cam = np.zeros((14, 14), np.uint8)
    cam[:, 7:] = 255
    assert (d.cam_overlay(img, cam, alpha=0.0, size=32) == 0).all()     # alpha 0: solo la imagen
    hot = d.cam_overlay(img, cam, alpha=1.0, size=32)
    assert hot.shape == (32, 32, 3) and hot[16, 30, 0] > hot[16, 2, 0]   # cálido a la derecha


def test_thin_epochs():
    assert d.thin_epochs([1, 2, 3], 8) == [1, 2, 3]
    out = d.thin_epochs(list(range(1, 51)), 8)
    assert len(out) <= 8 and out[0] == 1 and out[-1] == 50


def test_discovery_and_resolution(tmp_path):
    folder = make_inspector_run(tmp_path, ts="25102026_120000")
    make_inspector_run(tmp_path, ts="24102026_100000", with_events=False)
    make_inspector_run(tmp_path, ts="23102026_100000", with_inspect=False)
    runs = d.discover_inspector_runs(tmp_path)
    by_ts = {r.timestamp: r for r in runs}
    assert set(by_ts) == {"25102026_120000", "24102026_100000", "23102026_100000"}
    assert by_ts["25102026_120000"].events_path and by_ts["25102026_120000"].inspect_dir
    assert by_ts["24102026_100000"].events_path is None and by_ts["24102026_100000"].inspect_dir
    assert by_ts["23102026_100000"].inspect_dir is None
    assert runs[0].timestamp == "25102026_120000"                      # más reciente primero
    ms = {"dir": "inspect_25102026_120000"}
    assert d.resolve_inspect_dir(folder / "events_25102026_120000.jsonl", ms) == \
        folder / "inspect_25102026_120000"
    assert d.resolve_inspect_dir(folder / "events_x.jsonl", {"dir": "nope"}) is None
    assert d.resolve_inspect_dir(None, ms) is None


def test_reader_recognises_inspection_events(tmp_path):
    folder = make_inspector_run(tmp_path, epochs=2)
    f = EventsFollower(folder / "events_25102026_120000.jsonl")
    f.poll()
    s = f.state
    assert s.n_ignored == 0
    assert s.monitor_start["dir"] == "inspect_25102026_120000"
    assert len(s.systems) == 8 and len(s.layers) == 8
    assert sorted(s.histograms) == [1, 2] and len(s.inspections) == 2
    perf = d.batch_performance(s.batch_rows())
    assert {"samples_per_s", "data_frac"} <= set(perf.columns)


def test_reader_bounds_inspection_buffers():
    s = RunState()
    for i in range(6000):
        s.apply({"type": "system", "time": float(i)})
    for e in range(250):
        s.apply({"type": "histograms", "epoch": e, "groups": []})
    s.apply({"type": "histograms", "groups": []})          # sin época: se ignora sin fallar
    assert len(s.systems) == 5000 and len(s.histograms) == 200 and min(s.histograms) == 50


# ── Alertas (contrato del evento `alert`) y histogramas con `kind` ───────────────

def _alert(t, level, code, epoch=None, **kw):
    return {"type": "alert", "time": t, "level": level, "code": code,
            "message": f"msg {code}", "epoch": epoch, "batch": None, "value": 1.5,
            "threshold": 1.0, **kw}


def test_reader_collects_alerts_bounded_and_histogram_kinds():
    s = RunState()
    for i in range(600):
        s.apply(_alert(float(i), "warning", "overfitting", epoch=i))
    assert len(s.alerts) == 500 and s.alerts[0]["epoch"] == 100 and s.n_ignored == 0
    s.apply({"type": "histograms", "epoch": 1, "groups": []})                       # sin kind
    s.apply({"type": "histograms", "epoch": 1, "kind": "weights", "groups": []})
    s.apply({"type": "histograms", "epoch": 1, "kind": "gradients", "groups": []})
    s.apply({"type": "histograms", "epoch": 2, "kind": "gradients", "groups": []})
    assert sorted(s.histograms) == [1] and sorted(s.grad_histograms) == [1, 2]
    s.apply({"type": "histograms", "epoch": 3, "kind": "activations", "groups": []})
    s.apply({"type": "histograms", "epoch": 3, "kind": ["x"], "groups": []})
    assert s.n_ignored == 2 and 3 not in s.histograms and 3 not in s.grad_histograms
    assert d.events_have(s)["histograms"] is True
    only_grad = RunState()
    only_grad.apply({"type": "histograms", "epoch": 1, "kind": "gradients", "groups": []})
    assert d.events_have(only_grad)["histograms"] is True


def test_alert_frame_is_chronological_and_tolerant():
    evs = [_alert(20.0, "critical", "nan_loss", epoch=2, batch=7),
           _alert(10.0, "warning", "gpu_temp_high"),
           {"type": "alert", "time": 30.0, "level": "weird", "code": "x"}]
    df = d.alert_frame(evs)
    assert df["code"].tolist() == ["gpu_temp_high", "nan_loss", "x"]
    assert df["level"].tolist() == ["warning", "critical", "warning"]   # desconocido: warning
    assert df["epoch"].isna().tolist() == [True, False, True] and df["batch"].iloc[1] == 7
    assert str(df["time"].dt.tz) == "UTC"
    assert d.alert_frame([]).empty and list(d.alert_frame([]).columns) == d.ALERT_COLUMNS


def test_active_alerts_latest_per_code_critical_first_with_window():
    evs = [_alert(1.0, "warning", "overfitting", epoch=1),
           _alert(2.0, "warning", "overfitting", epoch=4),
           _alert(3.0, "critical", "nan_loss", epoch=2),
           _alert(4.0, "warning", "gpu_temp_high", epoch=None),
           _alert(5.0, "warning", "dataloader_bound", epoch=5)]
    # en curso, época 5, ventana 1: nan_loss (ep 2) ya no está activa
    live = d.active_alerts(evs, current_epoch=5)
    assert [a["code"] for a in live] == ["dataloader_bound", "gpu_temp_high", "overfitting"]
    assert [a["epoch"] for a in live if a["code"] == "overfitting"] == [4]
    # run terminado: todos los códigos, el crítico primero
    done = d.active_alerts(evs)
    assert done[0]["code"] == "nan_loss" and done[0]["level"] == "critical" and len(done) == 4
    assert d.active_alerts([], current_epoch=3) == []


def test_active_alerts_are_per_code_and_group():
    evs = [_alert(1.0, "warning", "grad_explosion", epoch=2, group="blocks.0"),
           _alert(2.0, "warning", "grad_explosion", epoch=2, group="blocks.1"),
           _alert(3.0, "warning", "grad_explosion", epoch=2, group="blocks.0")]
    act = d.active_alerts(evs, current_epoch=2)
    assert sorted(a["group"] for a in act) == ["blocks.0", "blocks.1"]
    assert d.alert_line(act[0]).startswith("epoch 2 · grad_explosion [blocks.")
    assert d.alert_frame(evs)["group"].tolist() == ["blocks.0", "blocks.1", "blocks.0"]
    assert d.alert_frame([_alert(1.0, "warning", "x", value=None)])["group"].tolist() == [""]


def test_alert_line():
    assert d.alert_line(_alert(1.0, "critical", "nan_loss", epoch=3)) == \
        "epoch 3 · nan_loss · msg nan_loss"
    assert d.alert_line({"code": "gpu_temp_high"}) == "gpu_temp_high"
