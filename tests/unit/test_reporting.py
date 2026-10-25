"""Informe HTML de un run (src/reporting/, ``paravit report``): resolución del argumento,
generación con los fixtures para las tres tareas, HTML válido y autocontenido, degradación sin
datos y lista TRIPOD+AI."""
import datetime as dt
import json
from html.parser import HTMLParser

import pytest
from typer.testing import CliRunner

from src.cli import app
from src.reporting import (NoInspection, RunNotFound, build_report, load_run, resolve_run,
                           write_report)
from src.reporting import tripod
from tests.fixtures.make_inspector_run import make_inspector_run

TS = "25102026_120000"
VOID = {"meta", "br", "img", "hr", "input", "link"}
SECTION_IDS = ["sec-summary", "sec-training", "sec-evaluation", "sec-dynamics", "sec-tripod"]
NOW = dt.datetime(2026, 10, 25, 12, 0, tzinfo=dt.timezone.utc)


class Checker(HTMLParser):
    """Parses the document and records tags balance, ids and resource references."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack, self.errors, self.ids, self.refs, self.tags = [], [], set(), [], set()
        self.big_scripts = 0
        self._script_len = 0
        self._in_script = False

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for k in ("src", "href"):
            if k in a and a[k] is not None:
                self.refs.append((tag, k, a[k]))
        if tag not in VOID:
            self.stack.append(tag)
        if tag == "script":
            self._in_script, self._script_len = True, 0

    def handle_endtag(self, tag):
        if tag in VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> with stack {self.stack[-3:]}")
            return
        self.stack.pop()
        if tag == "script":
            self._in_script = False
            self.big_scripts += self._script_len > 1_000_000

    def handle_data(self, data):
        if self._in_script:
            self._script_len += len(data)


def check(html: str) -> Checker:
    c = Checker()
    c.feed(html)
    c.close()
    return c


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """One synthetic run (with inspection data) per task."""
    out = {}
    for task in ("multiclass", "multilabel", "binary"):
        root = tmp_path_factory.mktemp(task)
        folder = make_inspector_run(root, task=task)
        out[task] = (root, folder)
    return out


@pytest.fixture(scope="module")
def reports(runs):
    return {t: build_report(load_run(f / f"events_{TS}.jsonl"), n_boot=40, now=NOW)
            for t, (_, f) in runs.items()}


# ── Resolución de <run> ────────────────────────────────────────────────────────

def test_resolve_events_file_folder_log_and_inspect_dir(runs):
    root, folder = runs["multiclass"]
    events = folder / f"events_{TS}.jsonl"
    assert resolve_run(events) == events
    assert resolve_run(folder) == events                       # run folder
    assert resolve_run(folder / f"train_{TS}.log") == events   # another file of the run
    assert resolve_run(folder / f"inspect_{TS}") == events     # inspection directory
    assert resolve_run(root / "logs" / "local") == events      # searches below the folder


def test_resolve_stamp_under_root(runs):
    root, folder = runs["binary"]
    assert resolve_run(TS, root=root) == folder / f"events_{TS}.jsonl"


def test_resolve_stamp_errors(tmp_path):
    with pytest.raises(RunNotFound, match="no events_"):
        resolve_run(TS, root=tmp_path)
    make_inspector_run(tmp_path, model="m_a", with_inspect=False)
    make_inspector_run(tmp_path, model="m_b", with_inspect=False)
    with pytest.raises(RunNotFound, match="ambiguous"):
        resolve_run(TS, root=tmp_path)


def test_resolve_picks_newest_in_folder_and_rejects_garbage(tmp_path):
    folder = make_inspector_run(tmp_path, ts="24102026_235959", with_inspect=False)
    make_inspector_run(tmp_path, ts="03112026_080000", with_inspect=False)
    # 03/11/2026 is newer than 24/10/2026 even though "24..." sorts after "03..."
    assert resolve_run(folder).name == "events_03112026_080000.jsonl"
    with pytest.raises(RunNotFound):
        resolve_run("not-a-run", root=tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RunNotFound, match="no events_"):
        resolve_run(empty)


def test_load_run_ignores_partial_line_and_unknown_events(tmp_path):
    folder = make_inspector_run(tmp_path, with_inspect=False)
    path = folder / f"events_{TS}.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"v": 1, "type": "future_event", "time": 1.0}) + "\n")
        f.write(json.dumps({"v": 99, "type": "epoch_end", "time": 1.0}) + "\n")
        f.write('{"v": 1, "type": "epoch_en')                    # half-written last line
    run = load_run(path)
    assert len(run.epoch_ends) == 3 and run.run_end is not None


# ── Generación ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("task", ["multiclass", "multilabel", "binary"])
def test_report_is_valid_self_contained_html(reports, task):
    html = reports[task]
    c = check(html)
    assert c.errors == [] and c.stack == []
    assert html.startswith("<!DOCTYPE html>")
    assert {"h1", "h2", "table", "img", "nav", "details"} <= c.tags
    for sid in SECTION_IDS:
        assert sid in c.ids
    # the only external references are plain links (the DOI), never resources
    for tag, attr, value in c.refs:
        if value.startswith(("http://", "https://")):
            assert tag == "a" and attr == "href", (tag, attr, value)
        else:
            assert tag == "a" and value.startswith("#") or value.startswith("data:image/png")
    assert "<link" not in html and "@import" not in html and "url(http" not in html
    assert c.big_scripts == 1                      # plotly.js embedded exactly once
    assert html.count("data:image/png;base64,") >= 8      # thumbnails and Grad-CAM


@pytest.mark.parametrize("task", ["multiclass", "multilabel", "binary"])
def test_report_has_every_section_and_block(reports, task):
    html = reports[task]
    for text in ("1. Summary", "2. Training", "3. Evaluation", "4. Training dynamics",
                 "5. TRIPOD+AI checklist", "Metrics with confidence intervals",
                 "ROC and precision-recall", "Per-class performance", "Confusion matrix",
                 "Calibration", "Operating points", "Subgroups", "Worst errors",
                 "Grad-CAM panel", "Data cartography", "Embeddings", "random seed = 0",
                 "n_boot = 40", "GPU-00000000", "Full configuration", "Alerts"):
        assert text in html, text
    ids = check(html).ids
    for i in ("eval-metrics", "eval-curves", "eval-perclass", "eval-confusion",
              "eval-calibration", "eval-operating", "eval-errors", "env", "config"):
        assert i in ids
    assert "sensitivity &ge; 0.90" in html and "sensitivity &ge; 0.95" in html
    assert "10.1136/bmj-2023-078378" in html


def test_report_is_deterministic(runs):
    run = load_run(runs["multilabel"][1] / f"events_{TS}.jsonl")
    a = build_report(run, n_boot=30, seed=7, now=NOW)
    b = build_report(run, n_boot=30, seed=7, now=NOW)
    c = build_report(run, n_boot=30, seed=8, now=NOW)
    assert a == b and a != c and "random seed = 7" in a


def test_epoch_option(runs):
    run = load_run(runs["multiclass"][1] / f"events_{TS}.jsonl")
    assert "Epoch 1 &middot;" in build_report(run, epoch=1, n_boot=20, now=NOW)
    assert "Epoch 3 &middot;" in build_report(run, n_boot=20, now=NOW)
    with pytest.raises(NoInspection, match="not inspected"):
        build_report(run, epoch=9, n_boot=20, now=NOW)


def test_task_specific_confusion_reading(reports):
    assert "co-activation" in reports["multilabel"]
    assert "Binary task" in reports["binary"]
    assert "Rows: true class" in reports["multiclass"]


def test_calibration_slope_is_reported_when_fit_exists(runs):
    # epoch 1 of the fixture is noisy enough for a finite slope and intercept
    run = load_run(runs["multiclass"][1] / f"events_{TS}.jsonl")
    html = build_report(run, epoch=1, n_boot=20, now=NOW)
    assert "Calibration slope (SE)" in html
    assert "The calibration fit does not exist" not in html


# ── Degradación ────────────────────────────────────────────────────────────────

def test_without_inspection_data_says_which_flag(tmp_path):
    folder = make_inspector_run(tmp_path, with_inspect=False)
    html = build_report(load_run(folder / f"events_{TS}.jsonl"), now=NOW)
    c = check(html)
    assert c.errors == [] and c.stack == []
    assert html.count("--monitor") >= 2             # evaluation and dynamics both name the flag
    assert "Metrics with confidence intervals" not in html
    assert "2. Training" in html and "5. TRIPOD+AI" in html


def test_without_system_events(tmp_path):
    folder = make_inspector_run(tmp_path, with_inspect=False)
    path = folder / f"events_{TS}.jsonl"
    keep = [ln for ln in path.read_text(encoding="utf-8").splitlines()
            if json.loads(ln)["type"] != "system"]
    path.write_text("\n".join(keep) + "\n", encoding="utf-8")
    html = build_report(load_run(path), now=NOW)
    assert "monitor.system_interval_s" in html
    assert "Approx. GPU energy" in html and "n/a (no system events" in html


def test_alerts_are_listed(tmp_path):
    folder = make_inspector_run(tmp_path, with_inspect=False)
    path = folder / f"events_{TS}.jsonl"
    alert = {"v": 1, "type": "alert", "time": 1792231300.0, "level": "warning",
             "code": "io_bound", "message": "DataLoader wait <b>high</b>", "epoch": 2,
             "batch": 10, "value": 0.41, "threshold": 0.3}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")
    html = build_report(load_run(path), now=NOW)
    assert "io_bound" in html and "level-warning" in html
    assert "&lt;b&gt;high&lt;/b&gt;" in html and "<b>high</b>" not in html    # escaped
    assert check(html).errors == []


def test_unfinished_run_and_missing_config(tmp_path):
    folder = make_inspector_run(tmp_path, with_inspect=False, finished=False)
    html = build_report(load_run(folder / f"events_{TS}.jsonl"), now=NOW)
    assert "no run_end event" in html


# ── TRIPOD+AI ──────────────────────────────────────────────────────────────────

def test_checklist_has_the_52_published_rows_with_unique_ids():
    ids = [row[0] for row in tripod.CHECKLIST]
    assert len(ids) == len(set(ids)) == 52
    assert {"3a", "12g", "18f", "23a", "27c"} <= set(ids)
    assert ids[0] == "1" and ids[-1] == "27c"


def test_tripod_is_honest_without_data():
    items = tripod.assess(tripod.Context())
    assert {it.status for it in items} == {tripod.PENDING}
    assert all(it.anchor is None for it in items)


def test_tripod_with_a_full_run(runs):
    run = load_run(runs["multilabel"][1] / f"events_{TS}.jsonl")
    html = build_report(run, n_boot=20, now=NOW)
    ids = check(html).ids
    items = {it.id: it for it in tripod.assess(
        tripod.Context(has_start=True, has_config=True, has_eval=True, n_boot=20))}
    assert items["12e"].status == tripod.DOCUMENTED
    # subgroups are not covered, so 23a can never be fully documented from a run
    assert items["23a"].status == tripod.PARTIAL
    for human in ("2", "3a", "4", "17", "18a", "19", "25", "26"):
        assert items[human].status == tripod.PENDING
    counts = tripod.summary(list(items.values()))
    assert counts[tripod.PENDING] > counts[tripod.DOCUMENTED] + counts[tripod.PARTIAL]
    for it in items.values():              # every link points to an existing section
        assert it.anchor is None or it.anchor in ids
    assert "1 documented" in html


def test_tripod_without_inspection_marks_evaluation_items_pending():
    items = {it.id: it for it in tripod.assess(
        tripod.Context(has_start=True, has_config=True, has_eval=False))}
    assert items["12e"].status == items["23a"].status == tripod.PENDING
    assert "--monitor" in items["12e"].note


# ── CLI ────────────────────────────────────────────────────────────────────────

def test_cli_report_writes_next_to_events_and_honours_output(runs, tmp_path):
    _, folder = runs["multiclass"]
    runner = CliRunner()
    res = runner.invoke(app, ["report", str(folder), "--n-boot", "20", "--errors", "2"])
    assert res.exit_code == 0, res.output
    default = folder / f"report_{TS}.html"
    assert default.is_file() and "Run report" in default.read_text(encoding="utf-8")
    custom = tmp_path / "sub" / "mine.html"
    res = runner.invoke(app, ["report", str(folder / f"events_{TS}.jsonl"), "-o", str(custom),
                              "--n-boot", "20", "--epoch", "2"])
    assert res.exit_code == 0 and custom.is_file()
    assert "Epoch 2 &middot;" in custom.read_text(encoding="utf-8")


def test_cli_report_errors_exit_1(runs, tmp_path):
    runner = CliRunner()
    res = runner.invoke(app, ["report", "01012000_000000"])
    assert res.exit_code == 1
    _, folder = runs["binary"]
    res = runner.invoke(app, ["report", str(folder), "--epoch", "42", "-o",
                              str(tmp_path / "x.html")])
    assert res.exit_code == 1 and not (tmp_path / "x.html").exists()


def test_write_report_default_path(runs):
    _, folder = runs["binary"]
    out = write_report(folder, n_boot=20, now=NOW)
    assert out == folder / f"report_{TS}.html"
