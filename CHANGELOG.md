# Changelog

Cambios notables del proyecto. Formato basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/).

## [Unreleased]

### Cambiado
- Documentación para agentes estandarizada: `AGENTS.md` (instrucciones compartidas) + `CLAUDE.md`
  mínimo que lo importa. El antiguo `CLAUDE.md` (1300 líneas) se reparte en `docs/`:
  `arquitectura.md`, `infraestructura.md`, `resultados_tfg.md`, `historial_tfg.md` y
  `datasets/bigearthnet.md`, con un índice en `docs/README.md`.
- Corregidas en la documentación varias referencias obsoletas (PNGs que ya no se generan, carpetas
  `feasibility/` renombradas a `benchmark/`, `FeasibilityChecker` → `BenchmarkChecker`).

### Añadido
- `docs/roadmap.md`: plan de trabajo posterior al TFG.
- `CHANGELOG.md`.

## [v1.0-tfg] - 2026-07-12

Versión entregada como Trabajo de Fin de Grado (ULL, julio de 2026). Entre otras cosas incluye:

- Entrenamiento de ViT sobre BigEarthNet-S2 con una arquitectura de decoradores (Decorator +
  Template Method) común a cuatro estrategias: single-GPU, DDP, paralelismo de modelo y DDP
  heterogéneo GPU+CPU.
- Predictor analítico de tiempo, memoria, energía y coste (`paravit estimate`) y benchmark empírico
  (`paravit benchmark`).
- CLI `paravit` y dashboard Streamlit.
- Estudio de escalado en Kaggle 2×T4 y VERODE; mejor modelo con F1 macro de 0,620 en test.

El detalle completo está en [`docs/historial_tfg.md`](docs/historial_tfg.md) y los resultados
en [`docs/resultados_tfg.md`](docs/resultados_tfg.md).

[Unreleased]: https://github.com/alerguezrojas/tfg-distributed-transformers/compare/v1.0-tfg...HEAD
[v1.0-tfg]: https://github.com/alerguezrojas/tfg-distributed-transformers/releases/tag/v1.0-tfg
