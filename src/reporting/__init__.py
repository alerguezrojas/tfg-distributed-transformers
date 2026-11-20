"""Self-contained HTML report of a run (``paravit report``): reads logs, never trains.

``resolve_run`` finds the events file from a path or a stamp, ``load_run`` reads it,
``build_report`` returns the HTML and ``write_report`` does all three and writes the file.
Layout: ``data`` (resolution and events), ``evaluation`` (statistics), ``figures`` (plotly),
``tripod`` (checklist), ``render`` (HTML helpers) and ``report`` (sections and entry points).
"""
from src.reporting.data import RunData, RunNotFound, load_run, resolve_run
from src.reporting.evaluation import NoInspection
from src.reporting.report import build_report, default_output, write_report

__all__ = ["NoInspection", "RunData", "RunNotFound", "build_report", "default_output",
           "load_run", "resolve_run", "write_report"]
