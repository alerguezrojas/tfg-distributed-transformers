"""Feasibility analysis package — one responsibility per module (SRP).

Splits the former 1600-line scripts/check_feasibility.py into focused units.
``scripts/check_feasibility.py`` is now a thin CLI over this package.
"""
from src.feasibility.value_objects import (
    ModelInfo, HardwareInfo, CPUInfo, DiskInfo, DatasetProfile, DDPScenario,
    PerformancePrediction, BenchmarkResult, FeasibilityReport,
)
from src.feasibility.model_analyzer import ModelAnalyzer
from src.feasibility.probes import HardwareProbe, DiskProbe, DatasetProfiler
from src.feasibility.predictor import PerformancePredictor
from src.feasibility.ddp_optimizer import DDPOptimizer
from src.feasibility.benchmarker import Benchmarker
from src.feasibility.time_estimator import TimeEstimator
from src.feasibility.report_formatter import ReportFormatter
from src.feasibility.checker import FeasibilityChecker

__all__ = [
    "ModelInfo", "HardwareInfo", "CPUInfo", "DiskInfo", "DatasetProfile",
    "DDPScenario", "PerformancePrediction", "BenchmarkResult", "FeasibilityReport",
    "ModelAnalyzer", "HardwareProbe", "DiskProbe", "DatasetProfiler",
    "PerformancePredictor", "DDPOptimizer", "Benchmarker", "TimeEstimator",
    "ReportFormatter", "FeasibilityChecker",
]
