"""Integration tests for all web parsers against real log/CSV files."""
import pytest
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent


def _find_files(pattern):
    return list(ROOT.rglob(pattern))


class TestLogParser:
    def test_parses_all_logs(self):
        from src.web.log_parser import parse_log
        logs = _find_files("train_*.log")
        assert len(logs) > 0, "No training logs found"
        for log_path in logs:
            df = parse_log(log_path)
            assert isinstance(df, pd.DataFrame), f"Failed on {log_path}"
            assert "epoch" in df.columns
            assert "val_f1" in df.columns

    def test_simple_trace_produces_rows(self):
        from src.web.log_parser import parse_log
        simple_logs = [p for p in _find_files("train_*.log")
                       if "deep" not in p.name]
        if not simple_logs:
            pytest.skip("No simple trace logs found")
        log = simple_logs[0]
        df = parse_log(log)
        assert len(df) >= 1

    def test_energy_columns_parsed_when_present(self):
        from src.web.log_parser import parse_log
        logs = _find_files("train_*.log")
        for log_path in logs:
            text = log_path.read_text(errors="replace")
            if "[energy]" in text:
                df = parse_log(log_path)
                # energy columns should be present (may be NaN if not in CSV)
                assert "energy_eval_wh" in df.columns or "energy_eval_j" in df.columns
                break

    def test_deep_resumen_both_field_orders(self):
        """El parser deep debe extraer val_acc esté best ANTES o DESPUÉS de él.
        Hay dos formatos reales:
          A) val_f1=X  best=X  val_acc=X   (cluster)
          B) val_f1=X  val_acc=X  best=X   (local)
        Un regex posicional fijo dejaba uno de los dos a 0 epochs."""
        from src.web.log_parser import _parse_deep
        a = ("2026-05-07 10:56:58 [INFO ] [E001/30] ══ RESUMEN  "
             "train_loss=0.1657 train_f1=0.5865 train_acc=0.9295 | "
             "val_loss=0.1628 val_f1=0.6121 best=0.6121 val_acc=0.9306 | "
             "val_prec=0.7285 val_rec=0.5649 | time=6075s ETA=64h")
        b = ("2026-05-26 23:02:26 [INFO ] [E001/1] ══ RESUMEN  "
             "train_loss=0.3632 train_f1=0.3214 train_acc=0.8970 | "
             "val_loss=0.2414 val_f1=0.4049 val_acc=0.9090 best=0.4049 | "
             "val_prec=0.6354 val_rec=0.3636 | time=1780s ETA=0h")
        for txt, exp_f1, exp_acc in [(a, 0.6121, 0.9306), (b, 0.4049, 0.9090)]:
            df = _parse_deep(txt)
            assert len(df) == 1, f"no parseó la línea: {txt[:60]}"
            assert abs(df["val_f1"].iloc[0] - exp_f1) < 1e-6
            assert abs(df["val_acc"].iloc[0] - exp_acc) < 1e-6
            assert df["epoch_time"].iloc[0] in (6075.0, 1780.0)

    def test_real_deep_logs_parse_nonempty(self):
        """Los logs deep reales (train_deep_*.log) que completaron al menos un
        epoch deben parsear >0 filas (regresión del v1 cluster invisible)."""
        from src.web.log_parser import parse_log
        deep_logs = [p for p in _find_files("train_deep_*.log")]
        if not deep_logs:
            pytest.skip("No hay logs deep")
        parsed_any = False
        for log_path in deep_logs:
            if "RESUMEN" in log_path.read_text(errors="replace"):
                df = parse_log(log_path)
                assert len(df) > 0, f"deep log con RESUMEN parseó 0 filas: {log_path}"
                assert df["val_f1"].notna().any()
                parsed_any = True
        if not parsed_any:
            pytest.skip("Ningún log deep tiene RESUMEN")

    def test_energy_parsed_for_distributed_trainers(self):
        """Los regex [energy]/[timed] deben casar también DDPTrainer y
        HeterogeneousDDPTrainer, no sólo 'Trainer'. (Bug del demo distribuido:
        el panel de energía no aparecía porque el regex pedía 'Trainer' literal.)"""
        from src.web.log_parser import (
            _ENERGY_TRAIN, _ENERGY_EVAL, _TIMED_TRAIN, _TIMED_EVAL,
        )
        train_line = ("2026-06-04 15:00:35 [INFO ] [energy] "
                      "HeterogeneousDDPTrainer.train_epoch: 10891.4 J  "
                      "(3.02540 Wh)  potencia media 45.5 W")
        eval_line = ("2026-06-04 15:00:59 [INFO ] [energy] "
                     "DDPTrainer.eval_epoch: 1058.1 J  (0.29391 Wh)  "
                     "potencia media 45.0 W")
        timed_train = ("2026-06-04 15:00:35 [INFO ] [timed] "
                       "HeterogeneousDDPTrainer.train_epoch: 246.56s")
        timed_eval = ("2026-06-04 15:00:59 [INFO ] [timed] "
                      "DDPTrainer.eval_epoch: 24.24s")

        mt = _ENERGY_TRAIN.search(train_line)
        assert mt and float(mt.group(1)) == 10891.4 and float(mt.group(2)) == 45.5
        me = _ENERGY_EVAL.search(eval_line)
        assert me and float(me.group(2)) == 0.29391 and float(me.group(3)) == 45.0
        assert _TIMED_TRAIN.search(timed_train).group(1) == "246.56"
        assert _TIMED_EVAL.search(timed_eval).group(1) == "24.24"

    def test_energy_extracted_from_real_distributed_log(self):
        """Sobre el log real del demo heterogéneo, parse_log debe rellenar las
        columnas de energía con valores no nulos."""
        from src.web.log_parser import parse_log
        hetero_logs = [p for p in _find_files("train_*.log")
                       if "ddp_hetero" in str(p)]
        if not hetero_logs:
            pytest.skip("No hay logs de DDP heterogéneo")
        for log_path in hetero_logs:
            if "[energy]" not in log_path.read_text(errors="replace"):
                continue
            df = parse_log(log_path)
            assert df["energy_train_j"].notna().any(), f"sin energía en {log_path}"
            assert df["energy_eval_wh"].notna().any()
            assert df["power_train_w"].notna().any()
            break


class TestEpochMetricsCSV:
    def test_all_csvs_parseable(self):
        csvs = _find_files("epoch_metrics_*.csv")
        assert len(csvs) > 0, "No epoch_metrics CSVs found"
        non_empty = 0
        for p in csvs:
            df = pd.read_csv(p)
            assert "epoch" in df.columns
            assert "val_f1" in df.columns
            if len(df) >= 1:
                non_empty += 1
        assert non_empty >= 1, "All epoch_metrics CSVs are empty"

    def test_val_f1_in_valid_range(self):
        csvs = _find_files("epoch_metrics_*.csv")
        for p in csvs:
            df = pd.read_csv(p)
            if "val_f1" in df.columns:
                valid = df["val_f1"].dropna()
                assert (valid >= 0.0).all(), f"Negative F1 in {p}"
                assert (valid <= 1.0).all(), f"F1 > 1.0 in {p}"


class TestPerclassParser:
    def test_all_perclass_csvs_parseable(self):
        from src.web.perclass_parser import parse_perclass_csv
        csvs = _find_files("perclass_metrics_*.csv")
        assert len(csvs) > 0, "No perclass CSVs found"
        for p in csvs:
            df = parse_perclass_csv(p)
            assert isinstance(df, pd.DataFrame)
            assert "class_name" in df.columns
            assert "f1" in df.columns

    def test_f1_values_valid(self):
        from src.web.perclass_parser import parse_perclass_csv
        csvs = _find_files("perclass_metrics_*.csv")
        for p in csvs[:3]:
            df = parse_perclass_csv(p)
            valid = df["f1"].dropna()
            assert (valid >= 0.0).all()
            assert (valid <= 1.0).all()


class TestBatchParser:
    def test_all_batch_csvs_parseable(self):
        from src.web.batch_parser import parse_batch_csv
        csvs = _find_files("batch_metrics_*.csv")
        assert len(csvs) > 0, "No batch_metrics CSVs found"
        for p in csvs:
            df = parse_batch_csv(p)
            assert isinstance(df, pd.DataFrame)
            assert "running_loss" in df.columns

    def test_running_loss_positive(self):
        from src.web.batch_parser import parse_batch_csv
        csvs = _find_files("batch_metrics_*.csv")
        for p in csvs[:3]:
            df = parse_batch_csv(p)
            if not df.empty:
                assert (df["running_loss"].dropna() >= 0).all()


class TestBenchmarkParser:
    def test_all_benchmark_csvs_parseable(self):
        from src.web.benchmark_parser import parse_benchmark_csv
        csvs = _find_files("benchmark_*.csv")
        assert len(csvs) > 0, "No benchmark CSVs found"
        for p in csvs:
            meta, df = parse_benchmark_csv(p)
            assert isinstance(meta, dict)
            assert isinstance(df, pd.DataFrame)

    def test_meta_has_model_name(self):
        from src.web.benchmark_parser import parse_benchmark_csv
        csvs = _find_files("benchmark_*.csv")
        for p in csvs[:3]:
            meta, _ = parse_benchmark_csv(p)
            assert "model_name" in meta

    def test_benchmark_df_has_batch_size(self):
        from src.web.benchmark_parser import parse_benchmark_csv
        csvs = _find_files("benchmark_*.csv")
        for p in csvs[:3]:
            _, df = parse_benchmark_csv(p)
            if not df.empty:
                assert "batch_size" in df.columns


class TestConfusionMatrixParser:
    def test_all_confusion_csvs_parseable(self):
        from src.web.confusion_matrix_parser import parse_confusion_matrix_csv
        csvs = _find_files("confusion_matrix_*.csv")
        if not csvs:
            pytest.skip("No confusion matrix CSVs found")
        for p in csvs[:5]:
            df = parse_confusion_matrix_csv(p)
            assert isinstance(df, pd.DataFrame)
            assert "epoch" in df.columns

    def test_matrix_values_in_range(self):
        from src.web.confusion_matrix_parser import parse_confusion_matrix_csv, get_matrix_for_epoch
        csvs = _find_files("confusion_matrix_*.csv")
        if not csvs:
            pytest.skip("No confusion matrix CSVs found")
        df = parse_confusion_matrix_csv(csvs[0])
        if df.empty:
            pytest.skip("Empty confusion matrix CSV")
        epoch = df["epoch"].min()
        pivot = get_matrix_for_epoch(df, epoch)
        # Normalized confusion matrix values should be in [0, 1]
        values = pivot.values.flatten()
        assert (values >= 0).all()
        assert (values <= 1.0 + 1e-6).all()


class TestRunRegistry:
    def test_discovers_runs(self):
        from src.web.run_registry import discover_runs
        runs = discover_runs(ROOT)
        assert len(runs) > 0

    def test_run_has_required_attrs(self):
        from src.web.run_registry import discover_runs, RunInfo
        runs = discover_runs(ROOT)
        for run in runs[:5]:
            assert hasattr(run, "log_path")
            assert hasattr(run, "label")
            assert hasattr(run, "env")
            assert hasattr(run, "trace_mode")
            assert hasattr(run, "mode")
            assert hasattr(run, "model")
            assert run.log_path.exists()

    def test_run_label_not_empty(self):
        from src.web.run_registry import discover_runs
        runs = discover_runs(ROOT)
        for run in runs[:5]:
            assert run.label.strip() != ""

    def test_benchmark_csvs_discovered(self):
        from src.web.run_registry import discover_benchmark_csvs
        csvs = discover_benchmark_csvs(ROOT)
        assert len(csvs) > 0


class TestBenchmarkComparisonSizes:
    """La comparación estimación-vs-real debe usar el tamaño REAL del dataset
    (subset o completo), no asumir el full BigEarthNet de 237871 imágenes."""

    def _feas_df(self):
        return pd.DataFrame([{
            "batch_size": 96, "trace_mode": "simple",
            "s_per_batch_train": 0.16, "imgs_per_s_train": 600.0,
            "s_per_batch_eval": 0.05, "imgs_per_s_eval": 1800.0,
            "peak_vram_gb": 3.0, "avg_power_w": 150.0,
            "est_train_min_per_epoch": 0.1, "est_eval_min_per_epoch": 0.01,
            "est_total_min_per_epoch": 0.2, "optimizer_steps_per_epoch": 53,
        }])

    def _actual_df(self):
        return pd.DataFrame({
            "epoch": [1, 2, 3], "val_f1": [0.1, 0.2, 0.25],
            "epoch_time": [30.0, 20.0, 22.0],
            "time_train_s": [25.0, 16.0, 18.0], "time_eval_s": [5.0, 4.0, 4.0],
        })

    def test_uses_subset_size_from_meta(self):
        from src.web.benchmark_comparison import build_comparison
        meta = {"model_name": "vit_tiny", "n_train": 5000, "n_val": 1500}
        cmp = build_comparison(meta, self._feas_df(), self._actual_df(),
                               batch_size=96, trace_mode="simple")
        steps = next(r for r in cmp.rows if r.metric.startswith("Optimizer steps"))
        # ⌈5000/96⌉ = 53 — coincide con el estimado del CSV
        assert steps.actual == 53
        assert "5000" in steps.formula and "237871" not in steps.formula
        thr = next(r for r in cmp.rows if r.metric == "Train throughput")
        # actual = n_train / train_time_medio ≈ 5000/19.67 ≈ 254, NO 237871/t
        assert thr.actual is not None and thr.actual < 1000

    def test_falls_back_to_full_set_when_sizes_absent(self):
        from src.web.benchmark_comparison import build_comparison
        meta = {"model_name": "vit_tiny"}  # sin n_train (CSV antiguo)
        cmp = build_comparison(meta, self._feas_df(), self._actual_df(),
                               batch_size=96, trace_mode="simple")
        steps = next(r for r in cmp.rows if r.metric.startswith("Optimizer steps"))
        assert "237871" in steps.formula  # fallback al full set

    def test_sizes_roundtrip_through_parser(self, tmp_path):
        """benchmark escribe #sizes y el parser lo lee como n_train/n_val."""
        from src.web.benchmark_parser import parse_benchmark_csv
        csv = tmp_path / "feas.csv"
        csv.write_text(
            "#meta,model_name,total_params_M,flops_mflops,hardware_name,total_vram_gb,free_vram_gb\n"
            "#meta,vit_tiny,5.5,34.3,Tesla V100,34.0,34.0\n"
            "#sizes,n_train,n_val\n"
            "#sizes,5000,1500\n"
            "batch_size,trace_mode,imgs_per_s_train\n"
            "96,simple,600\n"
        )
        meta, df = parse_benchmark_csv(csv)
        assert meta.get("n_train") == 5000
        assert meta.get("n_val") == 1500

    def test_hetero_runs_classified_as_distributed(self):
        """El demo heterogéneo vive en logs/verode/ddp_hetero/ → su mode debe
        empezar por 'ddp' para que la pestaña Análisis DDP lo empareje contra
        un run single-GPU. (Bug: la pestaña filtraba mode=='ddp' exacto y dejaba
        fuera los runs heterogéneos.)"""
        from src.web.run_registry import discover_runs
        runs = discover_runs(ROOT)
        hetero = [r for r in runs if "ddp_hetero" in str(r.log_path)]
        if not hetero:
            pytest.skip("No hay runs heterogéneos en logs/")
        for r in hetero:
            assert r.mode == "ddp_hetero"
            assert r.mode.startswith("ddp"), (
                "el filtro de la pestaña Análisis DDP usa mode.startswith('ddp')"
            )
