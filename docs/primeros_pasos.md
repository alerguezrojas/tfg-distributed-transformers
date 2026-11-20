# Primeros pasos

Guía para incorporarse al proyecto sin conocerlo. Siguiéndola en orden, en una o dos horas tendrán
el entorno listo, un entrenamiento pequeño hecho y visto en la web, y sabrán dónde mirar para
todo lo demás. Las normas de trabajo (ramas, PR, commits) están en [CONTRIBUTING.md](../CONTRIBUTING.md).

## 1. Qué es y qué no es

**ParaViT-Lab** es un framework para entrenar Vision Transformers (los modelos de
[timm](https://github.com/huggingface/pytorch-image-models) y, con el registro de backbones, otros)
de forma distribuida y **predecir su coste** (tiempo, memoria, energía) antes de entrenar. Nació como
TFG y su versión entregada es la etiqueta `v1.0-tfg`. Desde octubre de 2026 es trabajo de
continuación con vistas a una publicación: soporte para varios datasets (BigEarthNet, MedMNIST y
PANDA, un dataset de biopsias de próstata) y un predictor analítico validado con más máquinas.

Lo que **no** es: no es un sistema de producción ni de diagnóstico clínico, y no es un servicio web
(la web solo lee los resultados guardados, nunca entrena). El estado exacto de lo que está hecho y
lo que viene está en [roadmap.md](roadmap.md), sección «Estado actual».

## 2. Instalar

Necesitan `git` y [`uv`](https://docs.astral.sh/uv/), que descarga solo la versión de Python
necesaria (3.12). Una GPU solo hace falta para entrenar en serio.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # Linux/macOS; en Windows, ver README
git clone https://github.com/alerguezrojas/tfg-distributed-transformers.git
cd tfg-distributed-transformers
uv sync                                             # crea .venv con las dependencias
uvx pre-commit install --hook-type pre-commit --hook-type commit-msg
```

El último comando instala los controles previos al commit (ruff, ficheros grandes, gitleaks para
que no se cuelen credenciales y la regla de commits sin atribución de IA).

Trampas conocidas:

- En el clúster VERODE el nodo de login no tiene AVX2 (se usa siempre `srun`) y `uv sync` instala
  torch para CUDA 13, incompatible con su driver: hay que reinstalar la rueda cu118. Detalle en
  [infraestructura.md](infraestructura.md).
- NCCL no admite dos procesos en la misma GPU y gloo no admite `ReduceOp.AVG`.
- Los tests y la CI usan torch en CPU: no asuman CUDA en ellos.
- Los checkpoints pesan (~0,5 GB por run con ViT-S) y se acumulan en `checkpoints/`; vigilen el
  disco y borren los de las pruebas.

## 3. Comprobar que todo funciona

```bash
uv run pytest -q                      # solo CPU, sin datos; algo más de un minuto (más de 1.200 tests)
uv run ruff check src scripts tests   # lo mismo que pasa la CI
uv run paravit.py --help              # train | estimate | benchmark | eval | runs | dashboard | report | menu
uv run paravit.py estimate --model vit_base_patch16_224 --gpu "Tesla T4" --strategy ddp --n-gpus 2
```

`estimate` es el predictor analítico: da tiempo, memoria y energía sin GPU ni datos. El repositorio
ya trae los logs de los runs reales, así que la web se puede explorar ya mismo:

```bash
uv run paravit.py dashboard           # http://localhost:8501
```

## 4. Primer entrenamiento, pequeño

El dataset más sencillo es **BreastMNIST** (ecografías de mama de 28 px, 546 imágenes de
entrenamiento, binario): pesa pocos MB y un run entero dura menos de un minuto en una
GPU. Se usa un ViT diminuto (`vit_tiny`).

```bash
# 1. Descargar el dataset (ficheros .npz oficiales de Zenodo, con comprobación de MD5)
uv run python scripts/download_medmnist.py --root ~/datasets/medmnist --datasets medmnist_breast

# 2. Entrenar 2 épocas. --data-root sustituye la ruta del YAML (que es la de la máquina del autor)
uv run paravit.py train --strategy single --config configs/medmnist_breast_local.yaml \
    --data-root ~/datasets/medmnist --epochs 2 --monitor
```

Notas:

- Con `--epochs 1` el calentamiento (`warmup_epochs`) se baja solo a 0 y se avisa por el log; un
  YAML que ya traiga `warmup_epochs >= epochs` sigue fallando en el validador.
- Sin GPU se entrena en CPU, más despacio (unos minutos con este dataset).
- `paravit train` acepta también `--seed` y `--data-root` (y los pasa al script de la estrategia).
  `uv run paravit.py train --config ... --dry-run` muestra siempre el comando exacto que ejecutaría.
- `--monitor` guarda las predicciones por imagen de cada época para el Inspector de la web.

Resultado:

```bash
uv run paravit.py runs                 # lista los runs, con el F1 de validación y de test
uv run paravit.py dashboard            # el nuevo run aparece en la web (Run results, Inspector)
```

Los artefactos quedan en dos sitios: `logs/local/medmnist_breast/single/vit_tiny_patch16_224/`
(log, CSV por época, eventos; es lo que lee la web y lo que se versiona en git) y
`checkpoints/local/medmnist_breast/single/vit_tiny_patch16_224/<DDMMYYYY_HHMMSS>/` con `best.pt` y
`last.pt` (nunca se versionan). Para evaluar en test una sola vez, al final:

```bash
uv run paravit.py eval --config configs/medmnist_breast_local.yaml \
    --checkpoint checkpoints/local/medmnist_breast/single/vit_tiny_patch16_224/<sello> \
    --split test --save-predictions-auto
```

Los logs de pruebas no se suben: añadan al commit solo los runs que vayan a documentar. Los
modos de entrenamiento (varias GPU, GPU+CPU, paralelismo de modelo) y todas las opciones están
en la [guía de uso](guia_de_uso.md); los configs disponibles, en
[configs/README.md](../configs/README.md).

## 5. Mapa mental del código

1. `paravit.py` → `src/cli.py`: CLI Typer; cada subcomando lanza un script de `scripts/`.
2. `scripts/train_*.py`: un punto de entrada por estrategia (una GPU, DDP, heterogéneo, modelo
   paralelo); construyen el entrenador con `TrainingSessionBuilder`.
3. `src/data/`: una ficha `DatasetSpec` por dataset, con registro por nombre (`data.name`).
4. `src/models/`: `TimmClassifier` (backbone + cabeza), optimizador LLRD, registro de backbones.
5. `src/training/`: `Trainer` y sus subclases por estrategia; **el bucle vive solo en
   `EpochController.fit()`**.
6. `src/training/tasks.py`: lo que depende del tipo de problema (pérdida, sigmoide o softmax,
   métricas, umbral).
7. `src/training/decorators/`: comportamiento transversal (trazas, monitor, gráficos) como
   decoradores, no como `if`s en el entrenador.
8. `src/performance_model.py`: el predictor analítico; es puro (sin torch ni I/O).
9. `src/analysis/` y `src/web/`: análisis NumPy de las predicciones y panel Streamlit, que solo
   lee `logs/`.
10. `tests/`: todo en CPU; `tests/integration/test_behaviour_golden.py` fija los números exactos
    del entrenamiento.

El detalle está en [arquitectura.md](arquitectura.md) y las invariantes de diseño que no deben
romperse, en [AGENTS.md](../AGENTS.md).

## 6. Dónde está el estado del proyecto

- [roadmap.md](roadmap.md), sección «Estado actual»: qué está hecho, qué está en curso, el siguiente
  paso y las decisiones pendientes. Es lo primero que se lee al retomar.
- [CHANGELOG.md](../CHANGELOG.md): qué ha cambiado y cuándo; los cambios sin publicar están en
  *Unreleased*.
- [historial_tfg.md](historial_tfg.md) y [resultados_tfg.md](resultados_tfg.md): histórico hasta la
  entrega del TFG. Se consultan, no se editan.

Al cerrar un bloque de trabajo propio, actualicen el CHANGELOG y, si afecta al plan, el roadmap.

## 7. Añadir un dataset o un modelo

- **Dataset:** una ficha `DatasetSpec` y un lector, sin tocar el entrenamiento. Pasos en
  [datasets/README.md](datasets/README.md), «Cómo añadir un dataset»; una ficha de documentación por
  dataset en `docs/datasets/`.
- **Modelo:** el modelo se elige con `model.name`: un nombre de timm, uno del registro de
  backbones (por ejemplo I-JEPA) o `hf:<repositorio>` de Hugging Face. Con `model.freeze_backbone`
  se hace sonda lineal en vez de ajuste fino. Detalle, limitaciones y cómo registrar un modelo en
  [modelos/README.md](modelos/README.md).

## 8. Entrenar en Kaggle

Kaggle da 2 GPU T4 gratuitas (unas 30 horas a la semana) y tiene PANDA ya montado, así que no hay que
descargar los 400 GB. Los kernels están en [`kaggle/`](../kaggle/README.md) y se lanzan con
`scripts/kaggle_push.sh`. Necesitan la CLI de Kaggle con su propio token en `~/.kaggle/` (**nunca en
el repositorio**; gitleaks lo vigila) y haber aceptado las reglas de la competición PANDA. Allí se
explica también cómo comprobar que la partición es la misma que la local (por `md5`).

## 9. Comparar modelos de forma justa

Lo que se publique tiene que resistir una revisión. Reglas del proyecto:

1. **Mismas particiones y mismas transformaciones para todos.** En PANDA, el `splits.csv` fijo
   ([ficha](datasets/panda.md)); no se rehace.
2. **Tres semillas** por modelo y configuración (`--seed 42/43/44`), y se compara media ± desviación.
   Con una sola semilla, diferencias de ±0,02 de QWK son ruido; ya nos pasó con la pérdida ordinal
   en PANDA. [`scripts/summarize_runs.py`](../scripts/summarize_runs.py) agrega los runs de varias
   semillas.
3. **La métrica principal del dataset:** QWK en PANDA, AUC en MedMNIST, F1 macro en BigEarthNet.
4. **En PANDA, además, los cortes clínicos** ISUP ≥ 1 e ISUP ≥ 2 (sensibilidad, especificidad y
   AUC), que salen de `eval --save-predictions-auto`, y el resultado por hospital (Radboud y
   Karolinska).
5. **Separar calibración y validación.** El umbral y los hiperparámetros se eligen con validación; el
   test se mira una vez al final. En el predictor, las constantes se calibran con unas medidas y se
   validan con otras distintas.
6. **No mirar el test para decidir** (qué época, qué umbral, qué variante). Si un número de test
   se ha usado para elegir, hay que decirlo.
7. Mismos recursos para los modelos que se comparan: resolución, épocas y, si se comparan a la vez
   sonda lineal y ajuste fino, ambos para cada modelo.
8. Anotar con cada resultado el origen y la **licencia** de los pesos.

## 10. Licencias de datos y pesos

El código es GPLv3, pero los datos y los pesos tienen las suyas:

- **PANDA:** CC BY-NC-SA 4.0 (solo uso no comercial; las obras derivadas se comparten con la misma
  licencia). Es un dataset médico: **no se suben al repositorio imágenes, metadatos ni
  identificadores**, solo métricas agregadas ([SECURITY.md](../SECURITY.md)).
- **I-JEPA** (pesos de Meta): CC BY-NC 4.0, solo uso no comercial.
- BigEarthNet y MedMNIST: ver su ficha en [datasets/](datasets/README.md).
- Antes de usar unos pesos nuevos, comprueben la licencia de los **pesos**, a menudo distinta de la
  del código, y anótenla en su ficha.

## 11. Trabajar con agentes de IA (opcional)

El repositorio está preparado para trabajar con asistentes de código, pero no es obligatorio. Las
instrucciones comunes a todos (Claude Code, Codex, Copilot, Cursor) están en
[AGENTS.md](../AGENTS.md). Para Claude Code hay, además:

- Agentes especializados en `.claude/agents/`: `explorador` (buscar), `implementador` (tareas
  acotadas), `documentador`, `investigador` (bibliografía) y `revisor` (revisión independiente).
- Skills en `.claude/skills/`: `/estado` (dónde está el proyecto), `/cerrar-bloque`, `/revisar-pr` y
  `/nuevo-dataset`.
- Hooks en `.claude/hooks/`: al abrir sesión muestran el «Estado actual», pasan ruff al editar y
  bloquean commits con atribución de IA.

Las reglas valen igual para el código escrito por una persona o por un agente: tests, revisión y
nada que dependa de un chat. Cómo se reparte el trabajo y qué garantías hay, en
[flujo_ia.md](flujo_ia.md).
