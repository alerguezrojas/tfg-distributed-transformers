# Auditoría de arquitectura y usabilidad (23/11/2026)

> Auditoría de solo lectura del repositorio (rama `develop` más el estudio del predictor del PR
> #248), hecha por el agente revisor. Objetivo: que ParaViT-Lab funcione igual en cuatro máquinas
> (la RTX 3060 Ti local, Kaggle con 2 × T4, VERODE con Slurm y V100, y la de un alumno nuevo que
> trabajará con otros modelos, p. ej. JEPA) y prepararlo para estrategias distribuidas nuevas.
> Cada hallazgo lleva gravedad, evidencia `fichero:línea` y una propuesta; lo marcado como
> "comprobado" se ejecutó en CPU. Las líneas citadas son las del 23/11/2026 y pueden moverse.
> El seguimiento de lo que se corrige está en [`docs/roadmap.md`](../roadmap.md).

## Resumen ejecutivo

El núcleo está bien diseñado: el bucle único en `EpochController`, los decoradores, `DatasetSpec` y
`Task`, el predictor puro y la web que no importa torch. Lo que falla es lo que rodea al núcleo:

- **entorno y configuración por máquina**: rutas absolutas, torch sin variantes por máquina, sin
  comando de diagnóstico;
- **lanzadores duplicados**: cuatro `scripts/train_*.py` que han ido divergiendo;
- **supuestos de BigEarthNet que sobreviven** en la web, en `estimate` y en el resumen de
  resultados.

Hay dos errores de comportamiento que conviene corregir antes que nada: el `device_ids` de DDP
multinodo y la atribución de los CSV de test por carpeta. Después, lo que más rinde es un **perfil
de máquina** más `paravit doctor`, y **un solo lanzador de entrenamiento** con un registro de
estrategias, que es también lo que más abarata añadir FSDP, ZeRO o pipeline.

## (a) Arquitectura y trabajo en varias máquinas

**A1. Alta — DDP multinodo usa el rank global como índice de GPU.**
`src/training/ddp_trainer.py:64` (`device_ids=[rank]`) frente a `scripts/train_ddp.py:103-106`
(`cuda:(local_rank % device_count)`); el builder recibe el rank global (`scripts/train_ddp.py:185`).
Con `--nnodes=2 --nproc_per_node=1` en VERODE, el rank 1 pide `device_ids=[1]` en un nodo con una
sola V100 y DDP falla. En Kaggle no se nota porque rank = local_rank. Propuesta:
`device_ids=[device.index]`, un test con rank ≠ local_rank y quitar el comentario de
`train_ddp.py:100-102`, que contradice la restricción de NCCL de `AGENTS.md`.

**A2. Alta — torch sin variantes por máquina; `uv run` deshace el arreglo de VERODE.**
`pyproject.toml:25` (`torch>=2.11.0`), `uv.lock` (torch 2.15.0 de PyPI, cu132),
`docs/infraestructura.md:44-48` (2.7.1+cu118 a mano). Cualquier `uv run` en VERODE vuelve a la
rueda del lock, y 2.7.1 queda fuera del rango declarado. Propuesta: extras excluyentes `cpu`,
`cu118`, `cu12x` con `[tool.uv.sources]` e índices de PyTorch (`[tool.uv] conflicts`) y un rango de
torch que admita la máquina más antigua que se quiera soportar (o la decisión explícita de
abandonarla). Mientras tanto, documentar `UV_NO_SYNC=1`.

**A3. Alta — sin diagnóstico del entorno; la caída a CPU es silenciosa.**
No hay `paravit doctor`; `scripts/train_single_gpu.py:141` elige CPU si no hay CUDA y solo lo
imprime; `environment_fingerprint()` (`src/training/events.py:106-125`) no recoge CUDA, driver ni
cuDNN. Propuesta: `paravit doctor` (Python, torch/CUDA, driver, GPUs y su entrada en `GPU_TABLE`,
AVX2, NVML, rutas de la config y si existen, espacio libre en `checkpoints/`, `env` resultante) y
que `train` falle si la config pide GPU y no la hay (o pida `--allow-cpu`).

**A4. Alta — configuración acoplada a la máquina.**
28 YAML con rutas `/media/alejandro/...` y `/home/bejeque/...`; `train_v3.yaml` y
`train_cluster_v3.yaml` solo difieren en rutas, lote y `env`; `configs/train_demo_ddp.yaml` se usa
en Kaggle con rutas y `env` de VERODE; `--data-root` solo cambia `data.root`
(`src/training/config_overrides.py:8-11`) y BigEarthNet lee `data.metadata` absoluto
(`src/data/bigearthnet.py:136`); el kernel de Kaggle reimplementa la superposición de configs
(`kaggle/panda_train/run.py:115-138`). Propuesta: separar **experimento** de **máquina**: perfiles
`configs/machines/{local,kaggle,verode}.yaml` (o `~/.config/paravit/machine.yaml`) con `data_dir`,
`checkpoint_dir`, `env`, `num_workers` y lote máximo; selección con `--machine` o
`PARAVIT_MACHINE`; rutas de datos relativas a `data_dir`; un `--set clave=valor` genérico que
sustituya a `deep_update`.

**A5. Media — `output.env` no identifica la máquina.**
El YAML fija `env: local` y el builder lo usa tal cual (`src/training/builder.py:267`): el alumno
nuevo escribiría en `logs/local/...`, mezclado con los del autor (filtro por entorno en
`src/web/app.py:126-128`). Propuesta: `env` del perfil de máquina y, por defecto, del host;
renombrar la clave `env` del manifiesto (huella del entorno, `src/run_manifest.py:26-27`) a
`host_env`.

**A6. Media — lógica de entrenamiento y evaluación en `scripts/`.**
`scripts/eval.py` tiene 18 funciones y los tests lo cargan con `spec_from_file_location`; igual
`select_ensemble.py` y `compare_predictions.py`. La línea `Configuración:` se construye cuatro
veces, distinta en cada script (`train_single_gpu.py:236-241`, `train_ddp.py:209-216`,
`train_model_parallel.py:152`, `train_heterogeneous_ddp.py:220`). Las opciones divergen
(`--resume` e `--inspect` solo en single; `--batch-size` solo en single y model-parallel);
`train_ddp.py:55` usa por defecto una config de VERODE y `train_ddp.py:98` da `KeyError` sin
torchrun. Propuesta: `src/training/launch.py` con un único `parse_args`, carga de config,
`DataLoader` y línea `Configuración:`, más un registro de estrategias (A7); scripts de 5 líneas; la
lógica de `eval.py` a `src/evaluation/`.

**A7. Media — añadir una estrategia obliga a tocar unos 8 sitios.**
`src/cli.py`, `src/cli_menu.py`, `src/training/builder.py`, un `scripts/train_*.py` nuevo,
`src/performance_model.py`, `src/web/tabs/benchmark/predict.py` y `validate.py`,
`src/training/checkpoints.py`: 21 ficheros nombran estrategias, y con dos grafías
(`model-parallel` en `train`, `model_parallel` en `estimate` y en las carpetas). Propuesta: un
`StrategySpec` (nombre, lanzador, envoltura del modelo, trainer, política de checkpoint)
registrado como `DatasetSpec`, con un único nombre canónico.

**A8. Baja — paquete no instalable.**
`pyproject.toml` sin `[build-system]` ni `[project.scripts]`; el paquete se llama `src`;
`sys.path.insert` en cada script y en `src/web/app.py`; nombre y descripción del proyecto aún del
TFG ("on BigEarthNet-S2"). Propuesta: paquete `paravit` con *entry point*
`paravit = paravit.cli:app`.

**A9. Baja — un test depende del entorno.**
`SystemProbe` lee `CUDA_VISIBLE_DEVICES` (`src/training/system_sampler.py:80`) y
`tests/unit/test_system_sampler.py:106-109` espera 2 GPU simuladas; con la variable definida (Slurm
con `--gres`, Kaggle) fallan 5 tests (comprobado). Propuesta:
`monkeypatch.delenv("CUDA_VISIBLE_DEVICES")` en un *fixture* de `conftest.py`.

## (b) Accesibilidad y usabilidad del terminal

**B1. Alta — atribución errónea de los resultados de test.**
`src/web/run_registry.py:227-236` asocia todos los `test_*.csv` de la carpeta a cada run y
`src/cli.py:576-580` toma el primero. Comprobado: los 4 runs de `logs/local/medmnist_breast/...`
muestran el mismo Test F1 (0,7158) aunque cada uno tiene su `test_<ts>.csv`;
`src/web/tabs/run/curves.py:35` enseña los cuatro bajo cada run. Propuesta: emparejar por sello
(`test_<ts>.csv`) y mostrar los que no lo tengan como "sin asignar".

**B2. Media — `paravit runs` y `summarize_runs` piensan en F1.**
`src/cli.py:562-585,603` ("Best Val F1"; para PANDA da `val_f1` 0,607 en vez del QWK);
`src/results_summary.py:18` no incluye `qwk` y toma la semilla de la línea `Configuración:`
(`:44`), que DDP no escribe (comprobado en
`logs/kaggle/panda/ddp/vit_small_patch16_224/train_20112026_163243.log`); `scripts/eval.py:475-478`
no guarda el QWK en el CSV. Propuesta: `RunInfo.main_metric` en la CLI, semilla del manifiesto,
`qwk` en el CSV y en `METRICS`, línea `Configuración:` unificada (A6). Son claves nuevas: el
contrato con la web no se rompe.

**B3. Media — `estimate` presupone BigEarthNet.**
`src/cli.py:289` (`--dataset full|subset` = 237.871/5.000) y `:294` (`num_classes` 19);
`predict_quality` siempre devuelve un valor (`src/performance_model.py:664-676`) y la CLI lo
imprime como "Expected Val F1" (`src/cli.py:396-404`) para cualquier dataset; no acepta `--config`
(modelo, `img_size`, tamaños de partición, lote). Propuesta: `estimate --config` y ningún prior de
calidad fuera de BigEarthNet (invariante "no predecir lo que no se puede contrastar").

**B4. Media — búsqueda difusa de GPU equivocada.**
`src/performance_model.py:163-173`. Comprobado: `gpu_spec("NVIDIA GeForce RTX 3060")` devuelve la
RTX 3060 Ti de 8 GB (la 3060 tiene 12 GB); "RTX 4070" y "NVIDIA L4" devuelven `None`. Un alumno
con una RTX 3060 recibiría OOM y lotes máximos falsos. Propuesta: coincidencia exacta tras
normalizar el nombre y, si no la hay, aviso y ficha derivada de `src/gpu_specs.py`.

**B5. Media — CLI y menú incompletos e incoherentes.**
`train` no expone `--batch-size`, `--resume` ni `--set`; la ayuda de `--precision` dice
"(single-GPU)" (`src/cli.py:259`) aunque vale para ddp y model-parallel; el menú solo pregunta la
precisión en single (`src/cli_menu.py:157-159`), no ofrece `--monitor`, `compare-preds`,
`select-ensemble` ni `report`, y en `eval` no pregunta `--data-root` ni `--save-predictions`;
`cli.py:13` cita `checkpoint_epoch_009.pt` y el docstring de `dashboard` lista secciones que ya no
existen; `prepare_panda`, `download_medmnist`, `summarize_runs` y `backfill_run_manifests` solo
existen como scripts.

**B6. Baja — idioma mezclado en una misma salida.**
`scripts/eval.py:430` ("Aggregate metrics") junto a `:437` ("QWK (cuadrático)") y `:421`
("ajustando el umbral"); los scripts de entrenamiento imprimen en español y la CLI en inglés.
Propuesta: salida de terminal en inglés, como la web, sin tocar las claves del contrato
(`Configuración:`, `RESUMEN`…).

**B7. Baja — rutas relativas al directorio de trabajo.**
`src/training/builder.py:279,376-377` escriben en `checkpoints/` y `logs/` relativos al cwd:
lanzando `python scripts/...` desde otra carpeta los runs no aparecen en la web. Propuesta: rutas
relativas a la raíz del repo o al perfil de máquina.

## (c) Interfaz web

**C1. Media — rutas y constantes del autor.**
`src/web/tabs/home.py:33-40`, `src/web/tabs/dataset.py:28-35`, `src/web/ui/helpers.py:127-128`; las
pestañas *Run results* importan clases y particiones de BigEarthNet
(`src/web/tabs/run/confusions.py:11`, `perclass.py:8,50`). Propuesta: rutas del perfil de máquina y
paneles por dataset registrados por nombre.

**C2. Media — "Estimate vs Benchmark vs Run" no es multi-dataset.**
`src/web/tabs/benchmark/validate.py:210-217` llama a `predict()` sin `img_size` ni `in_chans` (PANDA
a 512 px se estima con los FLOPs de 224) y compara el prior de F1 de BigEarthNet con el `val_f1` de
cualquier dataset (`:181`); `predict.py:47` anuncia "time +4 %" en Kaggle 2 × T4 cuando PANDA a
512 px midió −40 %; en *Estimate* solo se elige resolución con "Other timm model…"
(`predict.py:87-92`). Propuesta: alimentar el analítico con el manifiesto del run, quitar la
calidad fuera de BigEarthNet y sustituir los porcentajes fijos por la tabla de validación de
`performance_model.md`.

**C3. Media — lo que la haría una herramienta profesional.**
No hay enlaces directos a un run (`st.query_params`), ni vista de **conjuntos** (`select-ensemble` y
`compare-preds` solo en terminal), ni tabla de **test por dataset**, ni "predicción frente a medida"
por estrategia con `timings.csv` de Kaggle, ni métricas por rank en DDP (casilla de la fase 4).

**C4. Baja — escalabilidad de la barra lateral.**
`src/web/app.py:146-152` crea un `st.button` por run; con 121 runs va bien (descubrimiento en
0,12 s, comprobado) pero no escala a cientos. Propuesta: tabla con búsqueda o `st.dataframe` con
selección, y filtros por modelo y semilla.

**C5. Baja — coherencia y accesibilidad.**
Título "Training Dashboard" (`src/web/app.py:36`) en vez de ParaViT-Lab; `validate.py:247` usa una
paleta propia con verde y rojo fuera de `theme.py` (poco legible con daltonismo verde-rojo); el KPI
"GPU time" suma tiempo de reloj sin multiplicar por las GPU (`src/web/tabs/home.py:75`); código
muerto `model_explorer.py` (importa torch y timm) y `system_monitor.py`, que solo usa
`tests/run_web_tabs_check.py`.

## (d) Estrategias nuevas y predictor: qué facilita y qué dificulta

- **Facilita:** el bucle y los decoradores no dependen de la estrategia; `Trainer` tiene pocos
  métodos que sobrescribir (`train_epoch`, `eval_epoch`, `save_*`); `unwrap_model` y
  `strip_wrapper_prefix` ya separan el envoltorio de los pesos; `performance_model.py` es puro y
  fácil de ampliar.
- **Dificulta FSDP y ZeRO:** el builder crea el optimizador **antes** de envolver el modelo
  (`builder.py:218-242`; `DDPTrainer` envuelve en su `__init__`), y FSDP2 (`fully_shard`) lo
  necesita después; `DDPTrainer.save_checkpoint` solo corre en el rank 0 (`ddp_trainer.py:104-107`)
  y con FSDP o `ZeroRedundancyOptimizer` el `state_dict` es colectivo (se bloquearía); el modelo de
  memoria no recibe la estrategia (`performance_model.py:308-321`).
- **Dificulta el pipeline:** `ModelParallelViT` solo admite ViT de timm y dos etapas
  (`src/models/model_parallel.py:39-40`), y el predictor calcula la VRAM de model-parallel como si
  todo el modelo estuviera en una GPU (`performance_model.py:508`): marca OOM justo en el caso para
  el que existe la estrategia.
- **Requisito previo:** el `StrategySpec` (A7) con un gancho `wrap(model)`, el optimizador después
  de la envoltura, una política de checkpoint colectiva y un `sharding_factor` en la memoria del
  predictor. Detalle de cada estrategia en
  [`estrategias_distribuidas.md`](estrategias_distribuidas.md).

## Ejecución en clúster

No hay plantillas Slurm, Apptainer ni Docker en el repositorio (casilla abierta de la fase 4). El
multinodo se lanza a mano en cada nodo (`docs/infraestructura.md:94-151`) y la CLI no usa las
variables de Slurm (`SLURM_NODEID`, `SLURM_JOB_NODELIST`) ni el rendezvous `c10d`. Propuesta:
`paravit train --launcher slurm` (`torchrun --rdzv-backend=c10d --rdzv-endpoint=<primer nodo>`, el
mismo comando en todos los nodos), plantillas `cluster/slurm/{single,ddp_multinode}.sbatch` (que
sirvan también con `srun` mientras `sbatch` falle en VERODE) e imagen Apptainer con torch cu118
(resuelve también A2 en VERODE).

## Logs y artefactos

El contrato escala bien en lectura (0,12 s para 121 runs), pero `discover_runs` indexa por sello
(`run_registry.py:186`): dos runs del mismo segundo en máquinas distintas se pisan.
`events_*.jsonl` y `test_preds_*.npz` ni se versionan ni están en `.gitignore` (46 eventos sin
seguimiento). Kaggle no fija el entorno (`pip install -q timm nvidia-ml-py torchinfo` sin versiones,
`kaggle/panda_train/run.py:165`; rama `develop` sin fijar). Propuesta: clave de run `host+sello`,
reglas explícitas en `.gitignore`, versiones fijadas con `uv export` en los kernels y un
`paravit sync-logs` (rsync o zip) entre máquinas.

## Calidad y documentación

Los tests son rápidos y amplios (1.479: bucle, web con `AppTest`, contratos). Faltan un test de
que las cuatro estrategias escriben la misma línea `Configuración:` y otro de DDP con
rank ≠ local_rank (A1). En la CI, `uv sync --frozen` descarga primero las ruedas CUDA del lock (46
paquetes `nvidia-*`) y luego las cambia por las de CPU (`.github/workflows/ci.yml:40-44`); un
extra `cpu` (A2) lo evita.

Documentación desactualizada: `docs/infraestructura.md` (`git pull`/`git push origin main` en
contra del flujo de ramas; "5 secciones"; ruta de logs antigua), `docs/arquitectura.md` (5
secciones, sección Import inexistente, "≤8 runs" frente a los 12 del código),
`docs/guia_de_uso.md` ("vista opcional en español" inexistente), `AGENTS.md` (la lista de
subcomandos omite `report`, `compare-preds`, `select-ensemble` y `predict`),
`docs/primeros_pasos.md` (`eval` sin `--data-root`, que falla en una máquina nueva) y
`configs/README.md` (trata de "tú").

## Plan por fases

| Fase | Qué | Por qué primero | Esfuerzo |
|---|---|---|---|
| 0. Errores | A1, B1, B2 (QWK en el CSV, semilla y pérdida en DDP), B4, A9, documentación desactualizada, `.gitignore` | Fallos o datos erróneos que hoy afectan a resultados y a quien se incorpora | S |
| 1. Entorno y máquinas | Extras de torch por CUDA (A2), `paravit doctor` (A3), perfiles de máquina, `--set` y `env` por host (A4, A5), paquete instalable (A8) | Desbloquea al alumno, VERODE y Kaggle con el mismo repositorio | M |
| 2. Un solo lanzador | `src/training/launch.py`, `StrategySpec`, lógica de eval a `src/` (A6, A7), CLI y menú completos (B5), `estimate --config` (B3) | Elimina la divergencia entre estrategias y prepara la línea (d) | M |
| 3. Web profesional | Paneles por dataset sin rutas (C1), predicción frente a medida con el manifiesto (C2), conjuntos, test por dataset y enlaces directos (C3), selector escalable y paleta (C4, C5) | Depende de la fase 2 para leer configs y resultados homogéneos | M-L |
| 4. Clúster | `--launcher slurm` con rendezvous, plantillas y Apptainer | Necesita el perfil de máquina (1) y el lanzador único (2) | M |
| 5. Estrategias nuevas | Optimizador después de la envoltura, checkpoint colectivo, `sharding_factor` en el predictor; luego pipeline, FSDP y ZeRO, cada uno validado con medidas | El invariante del predictor exige medidas reales; sin la fase 2 cada estrategia cuesta unos 8 ficheros | L |

## Comprobado y correcto

`src/performance_model.py` sigue siendo puro (solo `math` y `dataclasses`); `src/analysis/` y
`src/reporting/` no importan torch; la web solo importa torch de forma perezosa para perfilar un
modelo fuera de la tabla; la importación de runs protege contra rutas fuera de `logs/`
(`src/web/run_import.py`); la escritura de checkpoints y manifiestos es atómica; los checkpoints de
DDP se guardan sin el prefijo `module.`; el descubrimiento de runs es rápido; la suite pasa en CPU
salvo el test que depende del entorno (A9).
