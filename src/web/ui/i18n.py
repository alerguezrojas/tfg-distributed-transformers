"""Optional Spanish view for the dashboard (English stays the default).

The UI is authored in English (the deliverable requirement). This module adds a
*global translation layer*: when Spanish is selected, it wraps Streamlit's
text-bearing methods so their label/text argument is looked up in an EN→ES
dictionary before rendering — no call site changes. Strings not in the
dictionary (dynamic f-strings, chart titles, identifiers) pass through unchanged,
so the Spanish view is "most of the static UI in Spanish".

Usage (in app.py, before any rendering):
    import src.web.ui.i18n as i18n
    lang = st.session_state.get("_lang", "en")
    i18n.install(lang)
"""
from __future__ import annotations

import re

import streamlit as st
from streamlit.delta_generator import DeltaGenerator

_HEADER_RE = re.compile(r"^(#{1,6}\s+)(.*)$")

# Methods whose first positional argument is user-facing text.
_TEXT_METHODS = [
    "markdown", "caption", "write", "header", "subheader", "title", "text",
    "button", "checkbox", "radio", "selectbox", "multiselect", "slider",
    "number_input", "text_input", "expander", "info", "warning", "success",
    "error", "metric", "form_submit_button", "download_button",
]

_ORIG_DG: dict = {}
_ORIG_ST: dict = {}


def _tr(s):
    """Translate a string, transparently handling markdown ``##``/``**`` wrappers
    around a known phrase (e.g. '## Project overview', '**Top 5 best classes**')."""
    if not isinstance(s, str):
        return s
    if s in TRANSLATIONS:
        return TRANSLATIONS[s]
    m = _HEADER_RE.match(s)                      # '## Heading'
    if m and m.group(2) in TRANSLATIONS:
        return m.group(1) + TRANSLATIONS[m.group(2)]
    if s.startswith("**") and s.endswith("**") and s[2:-2] in TRANSLATIONS:  # '**bold**'
        return "**" + TRANSLATIONS[s[2:-2]] + "**"
    return s


def _text_index(args) -> int:
    """Index of the text argument: 1 when bound as a class method (args[0] is the
    DeltaGenerator ``self``), else 0 for module-level ``st.*`` calls."""
    return 1 if (args and isinstance(args[0], DeltaGenerator)) else 0


def _wrap_first(orig):
    def wrapper(*args, **kwargs):
        i = _text_index(args)
        if len(args) > i and isinstance(args[i], str):
            args = args[:i] + (_tr(args[i]),) + args[i + 1:]
        return orig(*args, **kwargs)
    return wrapper


def _wrap_tabs(orig):
    def wrapper(*args, **kwargs):
        i = _text_index(args)
        if len(args) > i and isinstance(args[i], (list, tuple)):
            translated = [_tr(x) if isinstance(x, str) else x for x in args[i]]
            args = args[:i] + (translated,) + args[i + 1:]
        return orig(*args, **kwargs)
    return wrapper


def _capture() -> None:
    if _ORIG_DG:
        return
    for m in _TEXT_METHODS + ["tabs"]:
        if hasattr(DeltaGenerator, m):
            _ORIG_DG[m] = getattr(DeltaGenerator, m)
        if hasattr(st, m):
            _ORIG_ST[m] = getattr(st, m)


def _restore() -> None:
    for m, fn in _ORIG_DG.items():
        setattr(DeltaGenerator, m, fn)
    for m, fn in _ORIG_ST.items():
        setattr(st, m, fn)


def install(lang: str) -> None:
    """Translate the UI to ``lang`` ('es') or restore English ('en')."""
    _capture()
    _restore()                      # idempotent: always start from English
    if lang != "es":
        return
    for m, fn in _ORIG_DG.items():
        setattr(DeltaGenerator, m, _wrap_tabs(fn) if m == "tabs" else _wrap_first(fn))
    for m, fn in _ORIG_ST.items():
        setattr(st, m, _wrap_tabs(fn) if m == "tabs" else _wrap_first(fn))


# ── EN → ES dictionary (static UI chrome) ───────────────────────────────────────

TRANSLATIONS: dict[str, str] = {
    # Sidebar navigation (groups + items)
    "ANALYZE": "ANALIZAR", "PLAN": "PLANIFICAR", "DATA & OPS": "DATOS Y OPERACIÓN",
    "Overview": "Resumen", "Run results": "Resultados del run", "Compare": "Comparar",
    # Top tabs + sub-tabs
    "Home": "Inicio", "Comparison": "Comparativa", "Feasibility": "Viabilidad",
    "Data & models": "Datos y modelos", "System": "Sistema",
    "Curves": "Curvas", "Per-class": "Por clase", "Time": "Tiempo", "Info": "Información",
    "Single vs Distributed": "Single vs Distribuido", "Overlay runs": "Superponer runs",
    "Models": "Modelos", "Live": "En vivo", "Launcher": "Lanzador",
    "Report": "Informe", "Prediction vs reality": "Predicción vs realidad",
    "Real study": "Estudio real", "Run analysis": "Ejecutar análisis",
    "Per epoch": "Por epoch", "Global history": "Historia global",
    "Trend": "Tendencia", "Confusion matrix": "Matriz de confusión",
    "DDP (multi-GPU)": "DDP (multi-GPU)",
    # Section headers
    "Project overview": "Vista general del proyecto", "All runs": "Todos los runs",
    "System status": "Estado del sistema", "Disk": "Disco",
    "Network (cumulative since boot)": "Red (acumulado desde el arranque)",
    "System monitor": "Monitor del sistema", "System profile": "Perfil del sistema",
    "Dataset I/O": "E/S del dataset", "Model memory": "Memoria del modelo",
    "Throughput benchmark": "Benchmark de throughput", "Time estimates": "Estimaciones de tiempo",
    "Data explorer — BigEarthNet-S2 v2.0": "Explorador de datos — BigEarthNet-S2 v2.0",
    "Dataset splits": "Splits del dataset",
    "Class distribution (train split)": "Distribución de clases (split de train)",
    "Class imbalance": "Desbalance de clases",
    "Example images per class": "Imágenes de ejemplo por clase",
    "Distribution by country (train)": "Distribución por país (train)",
    "Per-class difficulty vs frequency": "Dificultad por clase vs frecuencia",
    "Model explorer": "Explorador de modelos", "Parameters vs FLOPs": "Parámetros vs FLOPs",
    "Required VRAM by batch size": "VRAM requerida por batch size",
    "Quick launch": "Lanzamiento rápido",
    "DDP analysis — Single-GPU vs Distributed": "Análisis DDP — Single-GPU vs Distribuido",
    "DDP runs": "Runs DDP", "Speedup analysis": "Análisis de speedup",
    "Theoretical vs real scaling": "Escalado teórico vs real",
    "DDP analysis — Optimal resource distribution": "Análisis DDP — Distribución óptima de recursos",
    "Scenario table": "Tabla de escenarios", "Load distribution per GPU": "Distribución de carga por GPU",
    "Speedup: estimate vs scaling laws": "Speedup: estimación vs leyes de escalado",
    "Estimated total time per configuration": "Tiempo total estimado por configuración",
    "Empirical performance prediction": "Predicción empírica de rendimiento",
    "Estimated F1 curve": "Curva F1 estimada",
    "Feasibility prediction vs what actually happened": "Predicción del feasibility vs lo que pasó de verdad",
    "Empirical convergence study": "Estudio empírico de convergencia",
    "Measured convergence curve": "Curva de convergencia medida",
    "Energy consumption": "Consumo energético",
    "Metric radar at the best epoch": "Radar de métricas en el mejor epoch",
    "Precision — CUDA cores vs Tensor cores": "Precisión — núcleos CUDA vs Tensor cores",
    "A · On 1 GPU — estimated vs real time": "A · En 1 GPU — tiempo estimado vs real",
    "B · When distributing — predicted vs real speedup (2 GPUs)": "B · Al distribuir — speedup predicho vs real (2 GPUs)",
    # Metric labels
    "Total runs": "Total runs", "Best Val F1": "Mejor Val F1", "Top run": "Run destacado",
    "Total GPU time": "Tiempo GPU total", "Feasibility reports": "Reportes viabilidad",
    "Epochs completed": "Epochs completados", "Best epoch": "Mejor epoch", "Duration": "Duración",
    "F1 @ optimal threshold": "F1 @ threshold óptimo", "Model": "Modelo",
    "GPU utilization": "Utilización GPU", "Temperature": "Temperatura", "Usage": "Uso",
    "Logical cores": "Núcleos lógicos", "Physical cores": "Núcleos físicos", "Frequency": "Frecuencia",
    "Used": "Usada", "Total": "Total", "Available": "Disponible", "Usage %": "Uso %",
    "VRAM used": "VRAM usada", "VRAM total": "VRAM total", "Utilization": "Utilización",
    "Sent": "Enviado", "Received": "Recibido", "Architecture": "Arquitectura",
    "CUDA cores": "Núcleos CUDA", "Parameters (M)": "Parámetros (M)",
    "Total VRAM (GB)": "VRAM total (GB)", "Free RAM (GB)": "RAM libre (GB)", "Total RAM (GB)": "RAM total (GB)",
    "Disk type": "Tipo de disco", "Read speed": "Velocidad lectura",
    "I/O vs compute ratio": "Ratio E/S vs cómputo", "Weights (MB)": "Pesos (MB)",
    "Gradients (MB)": "Gradientes (MB)", "AdamW state (MB)": "Estado AdamW (MB)",
    "Activations/img (MB)": "Activaciones/img (MB)", "Total static (MB)": "Total estático (MB)",
    "Total duration": "Duración total", "Average/epoch": "Promedio/epoch",
    "Min/epoch": "Mínimo/epoch", "Max/epoch": "Máximo/epoch",
    "Estimated (min/epoch)": "Estimado (min/epoch)", "Real average (min/epoch)": "Real promedio (min/epoch)",
    "Relative error": "Error relativo", "Single-GPU runs": "Runs single-GPU",
    "Distributed runs": "Runs distribuidos", "Single-GPU epoch": "Epoch single-GPU",
    "Real speedup": "Speedup real", "Predicted speedup": "Speedup predicho",
    "Prediction error": "Error de predicción", "Estimated time/epoch": "Tiempo/epoch estimado",
    "Real time/epoch": "Tiempo/epoch real", "Real throughput": "Throughput real",
    "Expected Val F1": "Val F1 esperado", "Estimated best epoch": "Mejor epoch estimado",
    "Estimated early stop": "Early stop estimado", "Confidence": "Confianza",
    "Estimated Val F1": "Val F1 estimado", "Fit R²": "R² del ajuste", "Plateau (epoch)": "Plateau (epoch)",
    "Suggested LR": "LR sugerido", "Min-loss LR": "LR mín. loss", "Divergence LR": "LR divergencia",
    "Gradient norm": "Norma gradiente", "Suggested batch size": "Batch size sugerido",
    "Coeff. of variation": "Coef. variación", "Most frequent class": "Clase más frecuente",
    "Rarest class": "Clase más rara", "Imbalance ratio": "Ratio de desbalance",
    "Total patches": "Total patches", "Validation": "Validación", "Train": "Train",
    "Epochs recorded": "Epochs registrados", "Batches per epoch": "Batches por epoch",
    "Total records": "Total registros", "Initial LR": "LR inicial", "Final LR": "LR final",
    "Minimum LR": "LR mínimo", "Last Val F1": "Último Val F1", "Last Val Loss": "Último Val Loss",
    "Total eval energy": "Energía eval total", "Total train energy": "Energía train total",
    "Total energy": "Energía total", "FP32 (CUDA cores)": "FP32 (núcleos CUDA)",
    "Compute cap.": "Compute cap.",
    # Widget / button labels
    "Trace mode": "Modo de traza", "Live monitor": "Monitor en vivo",
    "Refresh interval (s)": "Intervalo de refresco (s)", "Active run": "Run activo",
    "Add a custom timm model": "Añadir modelo timm personalizado",
    "Anomaly detection": "Detección de anomalías", "Associated files": "Ficheros asociados",
    "Batch size for VRAM estimate": "Batch size para estimación VRAM", "Batch sizes": "Batch sizes",
    "Classes (max 8)": "Clases (máx 8)", "DDP run": "Run DDP", "DDP training": "Entrenamiento DDP",
    "Epochs for estimate": "Epochs para estimación", "Epochs for total estimate": "Epochs para estimación total",
    "Filter log lines": "Filtrar líneas del log", "Fn decorators": "Decoradores Fn",
    "Full epochs table": "Tabla de epochs completa", "GPU device": "Dispositivo GPU",
    "Inspect features": "Funciones de inspección", "Launch": "Lanzar", "Layers": "Capas",
    "Log tail": "Cola del log", "Metric": "Métrica", "Mode": "Modo",
    "Moving average (batches)": "Media móvil (batches)",
    "Moving average (global batches)": "Media móvil (batches globales)",
    "Override batch size": "Override batch size", "Override epochs": "Override epochs",
    "Raw data": "Datos en bruto", "Run feasibility analysis": "Ejecutar análisis de viabilidad",
    "Run metadata": "Metadatos del run", "See detail and formulas": "Ver detalle y fórmulas",
    "Single-GPU run": "Run single-GPU", "Single-GPU training": "Entrenamiento single-GPU",
    "Skip I/O measurement (faster)": "Omitir medición de E/S (más rápido)",
    "System refresh (s)": "Refresco sistema (s)", "Trace modes": "Modos de traza",
    "View anomalies": "Ver anomalías", "What to compare?": "¿Qué comparar?",
    "YAML config": "Config YAML", "Model families": "Familias de modelos",
    "Metrics to overlay": "Métricas a superponer",
    "Select runs to compare (max 4)": "Selecciona runs a comparar (máx 4)",
    "Global metric": "Métrica global", "Scaling laws to overlay": "Leyes de escalado a superponer",
    "Serial fraction s (Amdahl / Gustafson)": "Fracción serial s (Amdahl / Gustafson)",
    "Select a model to prefill the Launcher": "Selecciona modelo para prellenar el Lanzador",
    "Precision (Tensor-core switch)": "Precisión (interruptor Tensor cores)",
    "Compare FP32 vs Tensor cores": "Comparar FP32 vs Tensor cores",
    "Precision (Tensor cores)": "Precisión (Tensor cores)",
    "Mini-training steps": "Steps del mini-training",
    "Dataset path (optional — to measure real I/O)": "Ruta al dataset (opcional — para medir E/S real)",
    "NFS factor": "Factor NFS", "Mini-training steps ": "Steps del mini-training",
    "Details of the run selected in the sidebar.": "Detalle del run seleccionado en la barra lateral.",
    # Download buttons
    "Download": "Descargar", "Download benchmark": "Descargar benchmark",
    "Download DDP scenarios": "Descargar escenarios DDP", "Download estimates": "Descargar estimaciones",
    "Download full history": "Descargar historia completa",
    "Download model comparison": "Descargar comparativa de modelos",
    "Download predicted curve": "Descargar curva predicha", "Download runs table": "Descargar tabla de runs",
    "Download selected data": "Descargar datos seleccionados", "Download CSV": "Descargar CSV",
    "Download distribution": "Descargar distribución", "Download comparison": "Descargar comparativa",
    "Download per-class table": "Descargar tabla por clase", "Download time data": "Descargar datos de tiempo",
    # Short status / info messages
    "Analysis complete.": "Análisis completado.",
    "Could not determine the config for this run.": "No se pudo determinar la config de este run.",
    "Could not read disk usage.": "No se pudo leer el uso de disco.",
    "DDP training complete.": "Entrenamiento DDP completado.",
    "GPU info unavailable (nvidia-smi not found).": "Info GPU no disponible (nvidia-smi no encontrado).",
    "GPU: nvidia-smi unavailable": "GPU: nvidia-smi no disponible",
    "No anomalies detected": "Sin anomalías detectadas",
    "No anomalies detected in the log.": "Sin anomalías detectadas en el log.",
    "No GPU detected (nvidia-smi unavailable).": "No se detectó GPU (nvidia-smi no disponible).",
    "No metrics data for this run.": "No hay datos de métricas para este run.",
    "No per-class CSV for this run.": "Sin CSV de per-class para este run.",
    "No runs available.": "No hay runs disponibles.", "No runs found.": "No se encontraron runs.",
    "No runs found in logs/.": "No se encontraron runs en logs/.",
    "No runs with parseable metrics found.": "No se encontraron runs con métricas parseables.",
    "Run the feasibility analysis first.": "Ejecuta primero el análisis de viabilidad.",
    "Select a run in the sidebar.": "Selecciona un run en la barra lateral.",
    "Select at least 2 runs.": "Selecciona al menos 2 runs.",
    "Select at least one family.": "Selecciona al menos una familia.",
    "Select at least one batch size.": "Selecciona al menos un batch size.",
    "Training complete.": "Entrenamiento completado.",
    # Sub-headers / misc
    "Top 5 best classes": "Top 5 mejores clases", "Top 5 worst classes": "Top 5 peores clases",
    "Estimated vs Real": "Estimado vs Real",
}
