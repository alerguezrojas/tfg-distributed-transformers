"""Distributed Transformers — research dashboard (Dash + Mantine + Plotly).

A multi-page, interactive dashboard over the training artifacts in logs/. Run:

    uv run python dashboard/app.py     # → http://127.0.0.1:8050
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dash import Dash, dcc, html, callback, Input, Output, no_update
import dash_mantine_components as dmc
from dash_iconify import DashIconify

from dashboard import data as D
from dashboard import theme as T

app = Dash(__name__, suppress_callback_exceptions=True,
           external_stylesheets=dmc.styles.ALL, title="Distributed Transformers")

_RANKED = sorted([r for r in D.RUNS if r["best_f1"] is not None], key=lambda r: -r["best_f1"])
_TOP = _RANKED[0] if _RANKED else None
GRAPH_CFG = {"displayModeBar": False, "responsive": True}


# ── Shared UI ────────────────────────────────────────────────────────────────────
def kpi(label, value, sub=None):
    return dmc.Card(withBorder=True, radius="md", p="md", children=[
        dmc.Text(label, size="xs", c="dimmed", tt="uppercase", fw=600),
        dmc.Text(value, fw=650, mt=4, style={"fontFamily": T.MONO, "fontSize": "24px"}),
        dmc.Text(sub, size="xs", c="dimmed", mt=2) if sub else None,
    ])


def panel(title, *children, **kw):
    return dmc.Card(withBorder=True, radius="md", p="lg", children=[
        dmc.Text(title, size="sm", fw=600, mb="sm"), *children], **kw)


def page_head(title, sub):
    return html.Div([dmc.Title(title, order=1), dmc.Text(sub, c="dimmed", mt=2, mb="lg")])


def mode_badge(r):
    return dmc.Badge(r["mode_label"], variant="light", color=T.ACCENT, radius="sm")


def prec_badge(p):
    return dmc.Badge(p.upper(), variant="light",
                     color="orange" if p != "fp32" else "gray", radius="sm")


def graph(fid, fig=None, h=300):
    return dcc.Graph(id=fid, figure=fig or {}, config=GRAPH_CFG, style={"height": f"{h}px"})


# ── Figures (static builders) ──────────────────────────────────────────────────────
def best_f1_fig():
    tt = _RANKED[:8][::-1]
    colors = [T.GOOD if r["best_f1"] >= 0.6 else T.WARN if r["best_f1"] >= 0.4 else T.PALETTE[0] for r in tt]
    return T.barh_fig([f"{r['model']} · {r['date'][:5]} {r['date'][11:]}" for r in tt],
                      [r["best_f1"] for r in tt],
                      colors=colors, x_max=1, label_fmt=lambda v: f"{v:.3f}", h=300)


def strategy_fig():
    c = {}
    for r in D.RUNS:
        c[r["mode_label"]] = c.get(r["mode_label"], 0) + 1
    return T.donut_fig(list(c), list(c.values()), h=300)


def leaderboard():
    rows = []
    for r in _RANKED + [x for x in D.RUNS if x["best_f1"] is None]:
        rows.append(dmc.TableTr([
            dmc.TableTd(r["date"]), dmc.TableTd(r["model"]),
            dmc.TableTd(mode_badge(r)), dmc.TableTd(prec_badge(r["precision"])),
            dmc.TableTd(r["env"]), dmc.TableTd(str(r["epochs"])),
            dmc.TableTd(dmc.Group(gap="xs", wrap="nowrap", children=[
                dmc.Progress(value=(r["best_f1"] or 0) * 100, color=T.ACCENT, size="sm", w=90, radius="sm"),
                dmc.Text(f"{r['best_f1']:.3f}" if r["best_f1"] is not None else "—", size="sm", fw=600,
                         style={"fontFamily": T.MONO})])),
        ]))
    head = ["Run", "Model", "Strategy", "Precision", "Env", "Epochs", "Best Val F1"]
    return dmc.Table(highlightOnHover=True, verticalSpacing="sm", horizontalSpacing="md", children=[
        dmc.TableThead(dmc.TableTr([dmc.TableTh(h) for h in head])),
        dmc.TableTbody(rows)])


# ── Views ────────────────────────────────────────────────────────────────────────
def view_overview():
    hero = dmc.Card(radius="md", p="xl", mb="lg", style={
        "background": "linear-gradient(120deg,#2a4196,#3f59bd)", "color": "white", "border": "none"},
        children=[dmc.Group(justify="space-between", align="center", children=[
            dmc.Stack(gap=6, children=[
                dmc.Title("BigEarthNet-S2 · Vision Transformer", order=2, c="white"),
                dmc.Text("Single-GPU, distributed (DDP), model-parallel and heterogeneous training, "
                         "compared — with an analytic model that predicts speedup before you run it.",
                         size="sm", c="indigo.1", maw=580)]),
            dmc.Stack(gap=0, align="flex-end", children=[
                dmc.Text(f"{_TOP['best_f1']:.3f}" if _TOP else "—", c="white",
                         style={"fontFamily": T.MONO, "fontSize": "44px", "fontWeight": 650, "lineHeight": 1}),
                dmc.Text(f"Best Val F1 · {_TOP['model']}" if _TOP else "", size="xs", c="indigo.2", mt=6)])])])
    return html.Div([
        page_head("Overview", "The distributed-training study at a glance."),
        hero,
        dmc.SimpleGrid(cols={"base": 2, "md": 4}, mb="lg", children=[
            kpi("Runs", str(len(D.RUNS))),
            kpi("Best Val F1", f"{_TOP['best_f1']:.3f}" if _TOP else "—", _TOP["model"] if _TOP else None),
            kpi("Models", str(len({r["model"] for r in D.RUNS}))),
            kpi("Environments", str(len({r["env"] for r in D.RUNS}))),
        ]),
        dmc.SimpleGrid(cols={"base": 1, "lg": 2}, mb="lg", children=[
            panel("Best Val F1 by run (top 8)", graph("ov-bars", best_f1_fig())),
            panel("Runs by strategy", graph("ov-donut", strategy_fig())),
        ]),
        panel(f"All runs · {len(D.RUNS)}", html.Div(leaderboard(), style={"overflowX": "auto"})),
    ])


def view_run():
    runs = [r for r in D.RUNS if (r["curve"].get("val_f1") or [])]
    opts = [{"value": r["id"], "label": r["label"]} for r in runs]
    return html.Div([
        dmc.Group(justify="space-between", align="flex-end", mb="lg", children=[
            html.Div([dmc.Title("Run results", order=1),
                      dmc.Text("Curves and per-class metrics for one run.", c="dimmed", mt=2)]),
            dmc.Select(id="run-select", data=opts, value=opts[0]["value"] if opts else None,
                       w=320, searchable=True, allowDeselect=False)]),
        html.Div(id="run-badges", style={"marginBottom": "14px"}),
        dmc.SimpleGrid(id="run-kpis", cols={"base": 2, "md": 4}, mb="lg"),
        dmc.SimpleGrid(cols={"base": 1, "lg": 2}, mb="lg", children=[
            panel("F1 (macro)", graph("run-f1", h=280)),
            panel("Loss", graph("run-loss", h=280))]),
        panel("Per-class F1 — last epoch (sorted)", graph("run-pc", h=460)),
    ])


def view_compare():
    opts = [{"value": r["id"], "label": r["label"]} for r in D.RUNS]
    latest = D.RUNS[0] if D.RUNS else None
    default = [r["id"] for r in D.RUNS if latest and r["env"] == latest["env"]
               and r["id"][:8] == latest["id"][:8]][:6] or [o["value"] for o in opts[:2]]
    return html.Div([
        page_head("Compare", "Overlay any runs and see per-class differences."),
        panel("Select runs", dmc.MultiSelect(id="cmp-select", data=opts, value=default,
                                              searchable=True, maxValues=8, clearable=True)),
        dmc.Space(h="lg"),
        dmc.SimpleGrid(cols={"base": 1, "lg": 2}, mb="lg", children=[
            panel("Val F1 across epochs", graph("cmp-f1", h=300)),
            panel("Best Val F1", graph("cmp-bar", h=300))]),
        html.Div(id="cmp-dumbbell"),
    ])


def view_feasibility():
    f = D.FEAS
    if not f.get("scenarios"):
        return html.Div([page_head("Feasibility", ""), dmc.Text("Predictions unavailable.", c="dimmed")])
    cards = [dmc.Card(withBorder=True, radius="md", p="md", children=[
        dmc.Text(s["name"], size="xs", c="dimmed", tt="uppercase", fw=600),
        dmc.Text(f"{s['speedup']:.2f}×" if s["speedup"] is not None else "—", fw=650,
                 style={"fontFamily": T.MONO, "fontSize": "24px"}, mt=4),
        dmc.Text(f"{s['time']:.0f}s/ep · {s['vram']} GB · {s['bottleneck']}", size="xs", c="dimmed", mt=2),
    ]) for s in f["scenarios"]]
    val_rows = [dmc.TableTr([dmc.TableTd(v["q"]),
                dmc.TableTd(v["pred"], style={"fontFamily": T.MONO, "color": T.PALETTE[0]}),
                dmc.TableTd(v["real"], style={"fontFamily": T.MONO})]) for v in f["validation"]]
    return html.Div([
        page_head("Feasibility", "What the analytic model predicts — before running anything."),
        dmc.Group(mb="md", children=[
            dmc.Badge(f["model"], variant="light", color=T.ACCENT),
            dmc.Badge(f"on {f['gpu']}", variant="light", color="gray"),
            dmc.Text("closed-form prediction · subset 5000 · batch 96 · 15 epochs", size="sm", c="dimmed")]),
        dmc.SimpleGrid(cols={"base": 2, "md": 4}, mb="lg", children=cards),
        dmc.SimpleGrid(cols={"base": 1, "lg": 2}, children=[
            panel("DDP scaling — predicted speedup vs GPUs",
                  graph("fe-scale", T.scaling_fig([s["n"] for s in f["scaling"]], [s["speedup"] for s in f["scaling"]]))),
            panel("Predicted vs real · validation",
                  dmc.Table(children=[
                      dmc.TableThead(dmc.TableTr([dmc.TableTh("Quantity"), dmc.TableTh("Predicted"), dmc.TableTh("Real")])),
                      dmc.TableTbody(val_rows)]),
                  dmc.Alert("The analytic model reproduces the measured Kaggle 2×T4 results within a few percent. "
                            "Speedups are independent of its single calibration constant — genuine out-of-sample "
                            "predictions.", color=T.ACCENT, variant="light", mt="md", radius="md")),
        ]),
    ])


def view_dataset():
    ds = D.DATASET
    sp, cls = ds.get("splits", {}), ds.get("classes", [])
    return html.Div([
        page_head("Dataset", "BigEarthNet-S2 — splits, classes and the imbalance."),
        dmc.SimpleGrid(cols={"base": 2, "md": 4}, mb="lg", children=[
            kpi("Train", f"{sp.get('train', 0):,}"), kpi("Validation", f"{sp.get('val', 0):,}"),
            kpi("Test", f"{sp.get('test', 0):,}"), kpi("Classes", str(len(cls) or 19))]),
        dmc.SimpleGrid(cols={"base": 1, "lg": 2}, children=[
            panel("Class frequency — the imbalance that caps macro-F1",
                  graph("ds-tree", T.treemap_fig([c["cls"] for c in cls], [c["count"] for c in cls], h=340))),
            panel("Classes by train frequency",
                  graph("ds-bar", T.barh_fig(
                      [c["cls"] for c in sorted(cls, key=lambda x: x["count"])],
                      [c["count"] for c in sorted(cls, key=lambda x: x["count"])],
                      colors=T.PALETTE[0], h=340, left=210))),
        ]),
    ])


_VIEWS = {"/": view_overview, "/runs": view_run, "/compare": view_compare,
          "/feasibility": view_feasibility, "/dataset": view_dataset}
_NAV = [("/", "Overview", "tabler:layout-dashboard"), ("/runs", "Run results", "tabler:chart-histogram"),
        ("/compare", "Compare", "tabler:arrows-left-right"), ("/feasibility", "Feasibility", "tabler:gauge"),
        ("/dataset", "Dataset", "tabler:database")]


def nav_links(path):
    return [dmc.NavLink(label=lbl, href=href, active=(path == href), variant="filled",
                        leftSection=DashIconify(icon=ic, width=18), mb=2) for href, lbl, ic in _NAV]


# ── Callbacks ──────────────────────────────────────────────────────────────────────
@callback(Output("page-content", "children"), Output("navlinks", "children"), Input("url", "pathname"))
def route(path):
    path = path or "/"
    return _VIEWS.get(path, view_overview)(), nav_links(path)


@callback(Output("run-kpis", "children"), Output("run-badges", "children"),
          Output("run-f1", "figure"), Output("run-loss", "figure"), Output("run-pc", "figure"),
          Input("run-select", "value"))
def update_run(run_id):
    r = D.BY_ID.get(run_id)
    if not r:
        return no_update, no_update, no_update, no_update, no_update
    cu = r["curve"]
    kpis = [kpi("Best Val F1", f"{r['best_f1']:.3f}" if r["best_f1"] is not None else "—"),
            kpi("Best epoch", str(r["best_epoch"] or "—")), kpi("Epochs", str(r["epochs"])),
            kpi("Duration", f"{r['duration_min']} min" if r["duration_min"] is not None else "—")]
    badges = dmc.Group([mode_badge(r), prec_badge(r["precision"]),
                        dmc.Badge(r["env"], variant="light", color="gray"),
                        dmc.Badge(r["model"], variant="light", color="gray")])
    f1 = T.line_fig(cu.get("epoch", []), [
        {"name": "Train", "y": cu.get("train_f1"), "color": "#5c6470"},
        {"name": "Val", "y": cu.get("val_f1"), "color": T.PALETTE[0], "fill": True}], h=280)
    loss = T.line_fig(cu.get("epoch", []), [
        {"name": "Train", "y": cu.get("train_loss"), "color": "#5c6470"},
        {"name": "Val", "y": cu.get("val_loss"), "color": T.PALETTE[2]}], h=280)
    pc = r["perclass"]
    if pc:
        pcs = sorted(pc, key=lambda x: x["f1"])
        colors = [T.GOOD if p["f1"] >= 0.6 else T.WARN if p["f1"] >= 0.3 else T.BAD for p in pcs]
        pcfig = T.barh_fig([p["cls"] for p in pcs], [p["f1"] for p in pcs], colors=colors,
                           x_max=1, label_fmt=lambda v: f"{v:.2f}", h=max(360, len(pcs) * 22 + 60))
        pcfig.update_layout(margin=dict(l=210, r=46, t=14, b=34))
    else:
        pcfig = {}
    return kpis, badges, f1, loss, pcfig


@callback(Output("cmp-f1", "figure"), Output("cmp-bar", "figure"), Output("cmp-dumbbell", "children"),
          Input("cmp-select", "value"))
def update_compare(ids):
    sel = [D.BY_ID[i] for i in (ids or []) if i in D.BY_ID]
    if len(sel) < 2:
        return {}, {}, dmc.Text("Select at least 2 runs.", c="dimmed")
    eps = sorted({e for r in sel for e in (r["curve"].get("epoch") or [])})
    f1 = T.line_fig(eps, [{"name": _short(r), "y": r["curve"].get("val_f1"),
                           "color": T.PALETTE[i % len(T.PALETTE)]} for i, r in enumerate(sel)], h=300)
    sb = sorted([r for r in sel if r["best_f1"] is not None], key=lambda r: r["best_f1"])
    bar = T.barh_fig([_short(r) for r in sb], [r["best_f1"] for r in sb],
                     colors=[T.PALETTE[i % len(T.PALETTE)] for i in range(len(sb))],
                     x_max=1, label_fmt=lambda v: f"{v:.3f}", h=300)
    bar.update_layout(margin=dict(l=180, r=46, t=14, b=34))
    dumb = _dumbbell_panel(sel)
    return f1, bar, dumb


def _short(r):
    base = f"{r['model']} · {r['mode_label']}" + (f" {r['precision']}" if r["precision"] != "fp32" else "")
    return f"{base} · {r['date'][11:]}"   # time keeps labels unique (no merged bars)


def _dumbbell_panel(sel):
    wpc = [r for r in sel if r["perclass"]]
    if len(wpc) < 2:
        return None
    A, B = wpc[0], wpc[1]
    ma = {p["cls"]: p["f1"] for p in A["perclass"]}
    mb = {p["cls"]: p["f1"] for p in B["perclass"]}
    cls = sorted([c for c in ma if c in mb], key=lambda c: mb[c] - ma[c])
    better = sum(1 for c in cls if mb[c] - ma[c] > 0.01)
    worse = sum(1 for c in cls if mb[c] - ma[c] < -0.01)
    fig = T.dumbbell_fig(cls, [ma[c] for c in cls], [mb[c] for c in cls], _short(A), _short(B),
                         h=max(380, len(cls) * 24 + 80))
    avg = lambda m: sum(m.values()) / len(m) if m else 0
    return panel(f"Per-class F1 · {_short(A)} → {_short(B)}",
                 dcc.Graph(figure=fig, config=GRAPH_CFG),
                 dmc.Text(f"B improves {better} class(es) and loses {worse} — "
                          f"macro F1 {avg(ma):.3f} → {avg(mb):.3f}.", size="sm", c="dimmed", mt="sm"))


# ── Layout ───────────────────────────────────────────────────────────────────────
header = dmc.AppShellHeader(px="md", children=[dmc.Group(h="100%", justify="space-between", children=[
    dmc.Group(gap="sm", children=[
        DashIconify(icon="tabler:topology-star-ring-3", width=24, color="#4263eb"),
        dmc.Text("Distributed Transformers", fw=700),
        dmc.Badge("BigEarthNet-S2 · TFG", variant="light", color="gray")]),
    dmc.Anchor("GitHub", href="https://github.com/alerguezrojas/tfg-distributed-transformers",
               target="_blank", size="sm", c="dimmed")])])

navbar = dmc.AppShellNavbar(p="md", children=[html.Div(id="navlinks", children=nav_links("/"))])
main = dmc.AppShellMain(dmc.Container(id="page-content", fluid=True, children=view_overview()))

app.layout = dmc.MantineProvider(
    forceColorScheme="light", theme=T.MANTINE_THEME,
    children=html.Div([dcc.Location(id="url"), dmc.AppShell(
        header={"height": 56}, navbar={"width": 232, "breakpoint": "sm"}, padding="lg",
        children=[header, navbar, main])]))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8050, debug=False)
