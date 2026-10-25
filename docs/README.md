# Documentación

| Documento | Para qué |
|---|---|
| [guia_de_uso.md](guia_de_uso.md) | Manual de usuario: instalación, CLI, configs, estrategias, web, solución de problemas |
| [arquitectura.md](arquitectura.md) | Cómo está construido: modelo, decoradores, CLI, DDP, benchmark, artefactos, dashboard |
| [performance_model.md](performance_model.md) | Derivación y validación del predictor analítico (tiempo, memoria, energía) |
| [infraestructura.md](infraestructura.md) | Máquinas, clúster VERODE, lanzamiento por entorno, errores conocidos |
| [datasets/](datasets/) | Una ficha por dataset soportado ([BigEarthNet-S2](datasets/bigearthnet.md)) |
| [roadmap.md](roadmap.md) | Plan de trabajo posterior al TFG y su **estado actual** |
| [flujo_ia.md](flujo_ia.md) | Cómo se trabaja con IA: agentes, skills, hooks, permisos y controles |
| [eventos.md](eventos.md) | Esquema del flujo de eventos de cada run (`events_*.jsonl`): contrato entre el entrenamiento y la web |
| [research/observabilidad_entrenamiento.md](research/observabilidad_entrenamiento.md) | Seguimiento en vivo, formatos de eventos e interpretabilidad de ViT en imagen médica: qué usa el sector y qué encaja aquí |
| [research/estado_del_arte_prediccion.md](research/estado_del_arte_prediccion.md) | Trabajo relacionado sobre predicción del rendimiento del entrenamiento (tiempo, memoria, energía, escalado) y dónde encaja el predictor |
| [resultados_tfg.md](resultados_tfg.md) | Resultados experimentales hasta la entrega (histórico) |
| [historial_tfg.md](historial_tfg.md) | Bitácora del desarrollo hasta `v1.0-tfg` (congelada) |
| [class_diagram.svg](class_diagram.svg) | Diagrama de clases (fuente: `class_diagram.puml`) |

Runbooks operativos: [Kaggle 2×T4](kaggle_speedup_runbook.md) ·
[paralelismo de modelo](model_parallel_runbook.md) · [focal vs BCE en VERODE](verode_focal_runbook.md).

Los cambios entre versiones se registran en [`../CHANGELOG.md`](../CHANGELOG.md).
