"""Tests del orden cronológico de RunInfo.sort_key.

El formato DDMMYYYY no ordena cronológicamente como string crudo
('27052026' > '02062026' aunque mayo es anterior a junio). sort_key
normaliza a YYYYMMDD_HHMMSS para un orden correcto.
"""
from pathlib import Path

import pytest

from src.web.run_registry import RunInfo


def _run(ts: str) -> RunInfo:
    return RunInfo(timestamp=ts, log_path=Path(f"/tmp/train_{ts}.log"),
                   trace_mode="simple", env="local")


def test_sort_key_ddmmyyyy_format():
    """02062026 (2 jun) debe tener sort_key > 27052026 (27 may)."""
    run_jun = _run("02062026_201938")  # 2 junio 2026
    run_may = _run("27052026_233519")  # 27 mayo 2026
    assert run_jun.sort_key > run_may.sort_key, \
        "2 junio debe ordenar después de 27 mayo"


def test_sort_key_normalizes_to_yyyymmdd():
    run = _run("02062026_201938")
    assert run.sort_key == "20260602_201938"


def test_sort_key_legacy_yyyymmdd():
    """Formato legacy YYYYMMDD se mantiene tal cual."""
    run = _run("20260527_233519")
    assert run.sort_key == "20260527_233519"


def test_sort_key_chronological_order():
    """Una lista de runs se ordena cronológicamente con sort_key."""
    runs = [
        _run("27052026_233519"),  # 27 may
        _run("02062026_201938"),  # 2 jun  ← más reciente
        _run("13052026_161533"),  # 13 may ← más antiguo
        _run("28052026_100000"),  # 28 may
    ]
    ordered = sorted(runs, key=lambda r: r.sort_key, reverse=True)
    timestamps = [r.timestamp for r in ordered]
    assert timestamps == [
        "02062026_201938",  # 2 jun
        "28052026_100000",  # 28 may
        "27052026_233519",  # 27 may
        "13052026_161533",  # 13 may
    ], f"Orden incorrecto: {timestamps}"


def test_sort_key_same_day_orders_by_time():
    """Dos runs del mismo día se ordenan por hora."""
    morning = _run("02062026_090000")
    evening = _run("02062026_210000")
    assert evening.sort_key > morning.sort_key


def test_sort_key_mixed_formats():
    """Mezcla de formato legacy y nuevo se ordena correctamente."""
    legacy_may = _run("20260513_120000")   # 13 may (legacy)
    new_jun = _run("02062026_120000")      # 2 jun (nuevo)
    assert new_jun.sort_key > legacy_may.sort_key


def test_sort_key_does_not_crash_on_short_timestamp():
    """sort_key no debe crashear con timestamps malformados."""
    run = _run("0206")  # corto
    # No debe lanzar excepción
    _ = run.sort_key
