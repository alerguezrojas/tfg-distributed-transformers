"""HTML building blocks: escaping, tables, number formatting, page shell and style.

No template engine: the report is assembled from escaped fragments (``Raw`` marks the ones that
are already HTML). One stylesheet, system fonts, no external resource of any kind.
"""
from __future__ import annotations

import base64
import html
import io
import math
from typing import Any, Iterable

import numpy as np


class Raw(str):
    """A string that is already HTML (not escaped again)."""


def esc(x: Any) -> str:
    return x if isinstance(x, Raw) else html.escape(str(x), quote=True)


def num(x: Any, nd: int = 3) -> str:
    """Fixed-point number; ``n/a`` for None / NaN / inf."""
    if x is None or isinstance(x, bool):
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return esc(x)
    return f"{v:.{nd}f}" if math.isfinite(v) else "n/a"


def with_ci(est: Any, ci: Iterable[float], nd: int = 3) -> str:
    lo, hi = ci
    return f"{num(est, nd)} ({num(lo, nd)}, {num(hi, nd)})"


def duration(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "n/a"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min {s:02d} s"
    return f"{m} min {s:02d} s" if m else f"{s} s"


def table(headers: list[str], rows: list[list[Any]], num_cols: Iterable[int] = (),
          cls: str = "") -> Raw:
    numeric = set(num_cols)
    head = "".join(f"<th{' class=\"num\"' if i in numeric else ''}>{esc(h)}</th>"
                   for i, h in enumerate(headers))
    body = []
    for r in rows:
        cells = "".join(f"<td{' class=\"num\"' if i in numeric else ''}>{esc(c)}</td>"
                        for i, c in enumerate(r))
        body.append(f"<tr>{cells}</tr>")
    attr = f' class="{cls}"' if cls else ""
    return Raw(f"<table{attr}><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>")


def kv_table(pairs: list[tuple[str, Any]]) -> Raw:
    rows = "".join(f"<tr><th scope=\"row\">{esc(k)}</th><td>{esc(v)}</td></tr>" for k, v in pairs)
    return Raw(f"<table class=\"kv\"><tbody>{rows}</tbody></table>")


def note(text: Any, cls: str = "note") -> Raw:
    return Raw(f"<p class=\"{cls}\">{esc(text)}</p>")


def unavailable(what: str, how: str) -> Raw:
    """A section that cannot be filled: say what is missing and which flag brings it."""
    return Raw(f"<p class=\"missing\"><strong>Not available.</strong> {esc(what)} "
               f"<span class=\"how\">{esc(how)}</span></p>")


def png_data_uri(img: np.ndarray) -> str:
    """(h, w, 3) uint8 array -> data URI of a PNG."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def section(id_: str, title: str, *parts: Any, level: int = 2) -> Raw:
    inner = "".join(esc(p) for p in parts)
    return Raw(f"<section id=\"{id_}\"><h{level}>{esc(title)}</h{level}>{inner}</section>")


CSS = """
:root{--ink:#1f2933;--muted:#5f6b76;--line:#d9dee3;--soft:#f5f7f8;--accent:#0b5cad;
--ok:#1b6e4a;--warn:#8a6100;--todo:#6b7280;--bad:#a4262c}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:#fff;color:var(--ink);font:15px/1.55 system-ui,-apple-system,"Segoe UI",
Roboto,"Helvetica Neue",Arial,sans-serif}
main,header.top{max-width:1000px;margin:0 auto;padding:0 20px}
header.top{padding-top:32px;padding-bottom:12px;border-bottom:2px solid var(--ink)}
h1{font-size:24px;line-height:1.25;margin:0 0 4px;font-weight:650}
h2{font-size:19px;margin:40px 0 12px;padding-bottom:6px;border-bottom:1px solid var(--line);
font-weight:650}
h3{font-size:15.5px;margin:26px 0 8px;font-weight:650}
p{margin:8px 0}
a{color:var(--accent)}
.sub{color:var(--muted);margin:0}
nav.toc{margin:18px 0 0;padding:10px 14px;background:var(--soft);border:1px solid var(--line)}
nav.toc ol{margin:0;padding-left:20px;columns:2}
nav.toc li{margin:2px 0}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:13.5px}
th,td{padding:5px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
thead th{border-bottom:1.5px solid var(--ink);font-weight:600;white-space:nowrap}
.num{text-align:right;font-variant-numeric:tabular-nums}
table.kv th{width:210px;font-weight:600;color:var(--muted);white-space:nowrap}
table.kv{font-size:14px}
.scroll{overflow-x:auto}
.note{color:var(--muted);font-size:13.5px}
.missing{padding:9px 12px;border-left:3px solid var(--warn);background:var(--soft)}
.missing .how{color:var(--muted)}
.params{font-size:13.5px;color:var(--muted)}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px}
@media (max-width:760px){.grid2{grid-template-columns:1fr}nav.toc ol{columns:1}}
figure{margin:0}
figure.card{border:1px solid var(--line);padding:6px;background:#fff;font-size:12.5px}
figure.card img{width:100%;height:auto;display:block;image-rendering:pixelated}
figure.card figcaption{margin-top:5px;color:var(--ink)}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:4px}
details{margin:10px 0;border:1px solid var(--line)}
summary{cursor:pointer;padding:7px 12px;background:var(--soft);font-weight:600}
pre{margin:0;padding:12px;overflow:auto;font:12.5px/1.45 ui-monospace,SFMono-Regular,Menlo,
Consolas,monospace;max-height:520px}
.badge{display:inline-block;padding:0 7px;border:1px solid currentColor;border-radius:3px;
font-size:12px;font-weight:600;white-space:nowrap}
.documented{color:var(--ok)}.partial{color:var(--warn)}.pending{color:var(--todo)}
.level-error,.level-critical{color:var(--bad);font-weight:600}.level-warning{color:var(--warn);
font-weight:600}
footer{max-width:1000px;margin:48px auto 32px;padding:12px 20px 0;border-top:1px solid var(--line);
color:var(--muted);font-size:12.5px}
@media print{nav.toc{display:none}h2{break-after:avoid}figure,table,details{break-inside:avoid}}
"""


def page(title: str, subtitle: str, toc: list[tuple[str, str]], body: str, plotly_js: str,
         footer: str) -> str:
    """Complete self-contained document. ``plotly_js`` is embedded once."""
    items = "".join(f"<li><a href=\"#{i}\">{esc(t)}</a></li>" for i, t in toc)
    return (
        "<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        f"<title>{esc(title)}</title><style>{CSS}</style>"
        f"<script>{plotly_js}</script></head><body>"
        f"<header class=\"top\"><h1>{esc(title)}</h1><p class=\"sub\">{esc(subtitle)}</p>"
        f"<nav class=\"toc\" aria-label=\"Contents\"><ol>{items}</ol></nav></header>"
        f"<main>{body}</main><footer>{esc(footer)}</footer></body></html>\n")
