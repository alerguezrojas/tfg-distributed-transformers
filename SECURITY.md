# Política de seguridad

ParaViT-Lab es un proyecto académico de código abierto. No hay versiones con soporte formal: se
corrige sobre la rama `develop` y llega a `main` con el siguiente cierre de bloque.

## Cómo informar de una vulnerabilidad

Por favor, **no abran una incidencia pública**. Usen el aviso privado de seguridad de GitHub:
pestaña *Security* del repositorio, *Report a vulnerability*. Incluyan los pasos para reproducirlo
y la versión o el commit afectado. Se responderá en cuanto sea posible, sin plazos garantizados.

## Qué está en alcance

- El código de `src/` y `scripts/`: carga de checkpoints, lectura de configuración, ejecución de
  comandos y manejo de ficheros.
- El panel web (`src/web/`), en especial la importación de runs (zip o carpeta) y la lectura de
  `logs/`: no debe leer ni escribir fuera de `logs/`.
- Los flujos de CI (`.github/`).

Fuera de alcance: vulnerabilidades en dependencias de terceros que ya tengan aviso público (las
vigilan Dependabot y la auditoría de CI), y el uso del panel expuesto en una red no confiable (está
pensado para uso local y de solo lectura).

## Reglas del proyecto

- **Nada de credenciales en el repositorio:** ni tokens de Kaggle o Hugging Face ni claves de
  ningún tipo, tampoco en logs, configs o notebooks. Se pasan por entorno o por ficheros fuera
  del repo. Un hook de gitleaks (`.pre-commit-config.yaml`) lo comprueba antes de cada commit.
- **`torch.load` siempre con `weights_only=True`.** No se cargan pickles de origen desconocido;
  `np.load` va con `allow_pickle=False`.
- **YAML solo con `yaml.safe_load`**, y subprocesos con lista de argumentos (nunca `shell=True`
  con entrada externa).
- **Los datos médicos no se suben al repositorio** (imágenes, metadatos ni identificadores de
  pacientes). Solo se commitean métricas agregadas de los runs.
- Si se filtra una credencial, se revoca de inmediato en el servicio de origen; borrarla del
  historial de git no basta.

## Medidas automáticas

CodeQL (análisis estático), Dependabot (actualizaciones semanales), auditoría de dependencias con
`pip-audit` en CI y gitleaks en `pre-commit`.
