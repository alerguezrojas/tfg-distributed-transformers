<div align="center">

# Entrenamiento distribuido de modelos abiertos de aprendizaje automático basados en Transformers

**Trabajo de Fin de Grado · Grado en Ingeniería Informática · Universidad de La Laguna**

[![CI](https://github.com/alerguezrojas/tfg-distributed-transformers/actions/workflows/ci.yml/badge.svg)](https://github.com/alerguezrojas/tfg-distributed-transformers/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)

**Autor:** Alejandro Rodríguez Rojas · **Tutor:** Francisco Carmelo Almeida Rodríguez · **Cotutor:** Daniel Suárez Labena

</div>

---

## De qué va

Este TFG entrena un **Vision Transformer** para clasificar imágenes de satélite Sentinel-2
(**BigEarthNet-S2**, 19 clases de cobertura terrestre) y estudia, con medidas reales, **cómo escala el
entrenamiento al distribuirlo** entre varias GPU o entre GPU y CPU. Además del modelo, incluye un
**predictor** que estima tiempo, memoria y coste *sin entrenar*, un **análisis de viabilidad** que lo
valida con un *benchmark* real, y un **dashboard web** para visualizar y comparar resultados. Todo se
opera desde un único comando de terminal, `tfg`.

## Requisitos

- **git** y **[uv](https://docs.astral.sh/uv/)** (el gestor de paquetes de Python; instala Python por ti).
- Una **GPU NVIDIA** y el **dataset** solo hacen falta para *entrenar*. Para probar el proyecto
  (dashboard, predictor, pruebas) **basta con CPU**: el repositorio ya incluye los resultados reales.

## Puesta en marcha

```bash
# 1. Instalar uv (Linux/macOS; en Windows PowerShell: irm https://astral.sh/uv/install.ps1 | iex)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clonar el repositorio e instalar el entorno (uv crea .venv con todo lo necesario)
git clone https://github.com/alerguezrojas/tfg-distributed-transformers.git
cd tfg-distributed-transformers
uv sync

# 3. Abrir el menú interactivo
uv run tfg.py menu
```

## El menú (`tfg menu`)

Es la forma más sencilla de usar el proyecto: un **menú guiado** que pregunta los parámetros uno a uno,
sin necesidad de recordar ningún comando. Desde él puedes:

- **Predecir** una configuración (tiempo, memoria, coste y F1 esperada) **sin GPU**.
- **Entrenar** con la estrategia que quieras (1 GPU, varias GPU con DDP, paralelismo de modelo, GPU+CPU).
- Lanzar el **análisis de viabilidad** (*benchmark* real de la máquina).
- **Evaluar** un modelo en el conjunto de test.

Para ver y comparar los entrenamientos de forma visual, abre el **dashboard**:

```bash
uv run tfg.py dashboard          # → http://localhost:8501
```

> ¿Prefieres comandos sueltos? `uv run tfg.py --help` los lista todos. Cada uno admite `--dry-run` para
> ver el comando exacto sin ejecutarlo.

## Autoría

**Alejandro Rodríguez Rojas** — Grado en Ingeniería Informática, Universidad de La Laguna.
Tutor: **Francisco Carmelo Almeida Rodríguez**. Cotutor: **Daniel Suárez Labena**. Curso 2025/2026.
