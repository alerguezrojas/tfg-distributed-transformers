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


class TestFeasibilityParser:
    def test_all_feasibility_csvs_parseable(self):
        from src.web.feasibility_parser import parse_feasibility_csv
        csvs = _find_files("feasibility_*.csv")
        assert len(csvs) > 0, "No feasibility CSVs found"
        for p in csvs:
            meta, df = parse_feasibility_csv(p)
            assert isinstance(meta, dict)
            assert isinstance(df, pd.DataFrame)

    def test_meta_has_model_name(self):
        from src.web.feasibility_parser import parse_feasibility_csv
        csvs = _find_files("feasibility_*.csv")
        for p in csvs[:3]:
            meta, _ = parse_feasibility_csv(p)
            assert "model_name" in meta

    def test_benchmark_df_has_batch_size(self):
        from src.web.feasibility_parser import parse_feasibility_csv
        csvs = _find_files("feasibility_*.csv")
        for p in csvs[:3]:
            _, df = parse_feasibility_csv(p)
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

    def test_feasibility_csvs_discovered(self):
        from src.web.run_registry import discover_feasibility_csvs
        csvs = discover_feasibility_csvs(ROOT)
        assert len(csvs) > 0
