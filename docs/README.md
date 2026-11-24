# Documentación

| Documento | Para qué |
|---|---|
| [primeros_pasos.md](primeros_pasos.md) | Incorporación paso a paso: instalar, primer entrenamiento, mapa del código, protocolo para comparar modelos y licencias. Las normas de trabajo, en [CONTRIBUTING.md](../CONTRIBUTING.md) |
| [guia_de_uso.md](guia_de_uso.md) | Manual de usuario: instalación, CLI, configs, estrategias, web, solución de problemas |
| [arquitectura.md](arquitectura.md) | Cómo está construido: modelo, decoradores, CLI, DDP, benchmark, artefactos, dashboard |
| [performance_model.md](performance_model.md) | Derivación y validación del predictor analítico (tiempo, memoria, energía) |
| [infraestructura.md](infraestructura.md) | Máquinas, clúster VERODE, lanzamiento por entorno, errores conocidos |
| [datasets/](datasets/) | Una ficha por dataset soportado ([BigEarthNet-S2](datasets/bigearthnet.md), [MedMNIST](datasets/medmnist.md), [PANDA](datasets/panda.md)) |
| [modelos/](modelos/README.md) | Cómo se elige y se añade un modelo: timm, registro propio (`BackboneSpec`), `hf:` (I-JEPA), sonda lineal frente a ajuste fino, protocolo de comparación y licencias |
| [roadmap.md](roadmap.md) | Plan de trabajo posterior al TFG y su **estado actual** |
| [flujo_ia.md](flujo_ia.md) | Cómo se trabaja con IA: agentes, skills, hooks, permisos y controles |
| [SECURITY.md](../SECURITY.md) | Política de seguridad: cómo informar de una vulnerabilidad y reglas del proyecto (secretos, `torch.load`, datos médicos) |
| [inspeccion.md](inspeccion.md) | Qué captura la inspección del entrenamiento (sistema, rendimiento, predicciones por muestra, Grad-CAM, capas) y en qué formato |
| [eventos.md](eventos.md) | Esquema del flujo de eventos de cada run (`events_*.jsonl`): contrato entre el entrenamiento y la web |
| [research/inventario_herramientas_observabilidad.md](research/inventario_herramientas_observabilidad.md) | Inventario funcionalidad a funcionalidad de W&B, TensorBoard, Comet, Aim, ClearML, FiftyOne, Lightning y otras: qué muestran, matriz comparativa, requisitos de imagen médica y lista priorizada para el proyecto |
| [research/observabilidad_entrenamiento.md](research/observabilidad_entrenamiento.md) | Seguimiento en vivo, formatos de eventos e interpretabilidad de ViT en imagen médica: qué usa el sector y qué encaja aquí |
| [research/estado_del_arte_prediccion.md](research/estado_del_arte_prediccion.md) | Trabajo relacionado sobre predicción del rendimiento del entrenamiento (tiempo, memoria, energía, escalado) y dónde encaja el predictor |
| [research/auditoria_arquitectura_usabilidad.md](research/auditoria_arquitectura_usabilidad.md) | Auditoría (23/11/2026) de arquitectura, trabajo en varias máquinas, usabilidad del terminal y web: hallazgos con evidencia y plan por fases |
| [research/estrategias_distribuidas.md](research/estrategias_distribuidas.md) | Catálogo de estrategias distribuidas (DDP, ZeRO, FSDP, tensor, pipeline, checkpointing, offload), costes de comunicación y memoria, viabilidad en 2×T4 y qué términos añadir al predictor |
| [research/panda.md](research/panda.md) | Dataset PANDA (biopsias de próstata, ISUP 0-5): formato, licencia, métrica QWK, soluciones ganadoras, ViT, sesgo de centro y cómo encajarlo en el proyecto |
| [research/medicion_clinica_prostata.md](research/medicion_clinica_prostata.md) | Cómo se gradúa el cáncer de próstata (Gleason, ISUP, variabilidad entre patólogos), cómo se miden los modelos de IA (QWK, AUC, sensibilidad/especificidad, validación externa) y qué pedir a un dataset clínico nuevo |
| [research/jepa_y_modelos_base.md](research/jepa_y_modelos_base.md) | JEPA (I-JEPA, V-JEPA, V-JEPA 2) frente a MAE/DINO, pesos y licencias, modelos de base de patología en PANDA y qué hace falta para usar backbones que no son de timm |
| [research/entrenamiento_elastico.md](research/entrenamiento_elastico.md) | Entrenamiento elástico (torchrun, Horovod, DeepSpeed, TorchFT), trabajos que modelan rendimiento para asignar GPU (Optimus, Pollux, AFS, Sia), encaje en ParaViT-Lab y tres pasos propuestos |
| [resultados_tfg.md](resultados_tfg.md) | Resultados experimentales hasta la entrega (histórico) |
| [historial_tfg.md](historial_tfg.md) | Bitácora del desarrollo hasta `v1.0-tfg` (congelada) |
| [class_diagram.svg](class_diagram.svg) | Diagrama de clases (fuente: `class_diagram.puml`) |

Runbooks operativos: [Kaggle 2×T4](kaggle_speedup_runbook.md) ·
[paralelismo de modelo](model_parallel_runbook.md) · [focal vs BCE en VERODE](verode_focal_runbook.md).

Los cambios entre versiones se registran en [`../CHANGELOG.md`](../CHANGELOG.md).
