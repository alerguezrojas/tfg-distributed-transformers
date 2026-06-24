"""Benchmark analysis package — one responsibility per module (SRP).

Splits the former 1600-line scripts/benchmark.py into focused units.
``scripts/benchmark.py`` is now a thin CLI over this package.
"""
from src.benchmark.value_objects import (
    ModelInfo, HardwareInfo, CPUInfo, DiskInfo, DatasetProfile, DDPScenario,
    PerformancePrediction, BenchmarkResult, BenchmarkReport,
)
from src.benchmark.model_analyzer import ModelAnalyzer
from src.benchmark.probes import HardwareProbe, DiskProbe, DatasetProfiler
from src.benchmark.predictor import PerformancePredictor
from src.benchmark.ddp_optimizer import DDPOptimizer
from src.benchmark.benchmarker import Benchmarker
from src.benchmark.time_estimator import TimeEstimator
from src.benchmark.report_formatter import ReportFormatter
from src.benchmark.checker import BenchmarkChecker

__all__ = [
    "ModelInfo", "HardwareInfo", "CPUInfo", "DiskInfo", "DatasetProfile",
    "DDPScenario", "PerformancePrediction", "BenchmarkResult", "BenchmarkReport",
    "ModelAnalyzer", "HardwareProbe", "DiskProbe", "DatasetProfiler",
    "PerformancePredictor", "DDPOptimizer", "Benchmarker", "TimeEstimator",
    "ReportFormatter", "BenchmarkChecker",
]
