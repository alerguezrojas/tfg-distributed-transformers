---
name: cerrar-bloque
description: Cierra un bloque de trabajo de forma que el proyecto pueda retomarse desde otra máquina o sesión — verificación, CHANGELOG, roadmap con su "Estado actual" y documentación. Úsala al terminar una feature, un experimento o una decisión, sin esperar a que lo pidan.
---

Lista de cierre (no te saltes pasos; si uno no aplica, dilo):

1. **Verificación**: `uv run pytest -q` en verde y `uv run ruff check src scripts tests` sin avisos.
   Si se tocó el núcleo de entrenamiento, los tests de caracterización
   (`tests/integration/test_behaviour_golden.py`) deben pasar sin cambiar valores.
2. **CHANGELOG.md**: entrada en `[Unreleased]` (Añadido / Cambiado / Corregido).
3. **docs/roadmap.md**: marca las casillas hechas y reescribe la sección **"Estado actual"**:
   hecho, en curso (con rama o PR), siguiente paso y decisiones pendientes del autor o del tutor.
4. **Documentación afectada**: `AGENTS.md` (comandos, mapa del código, convenciones), `docs/`
   (arquitectura, datasets, resultados) y el diagrama de clases si cambiaron clases públicas.
5. **Resultados**: si hubo experimentos, que los logs/CSV estén en `logs/` y los números en su
   documento (`docs/datasets/*.md` o el de resultados), con fecha, configuración y semillas.
6. **Git**: commits en español (Conventional Commits, sin `Co-Authored-By`), PR a `develop`.
   `develop → main` solo con el visto bueno del autor.
7. **Memoria automática**: solo preferencias personales del autor; nada imprescindible para
   continuar el proyecto puede quedar únicamente ahí.

Termina con un resumen de tres líneas para el autor.
