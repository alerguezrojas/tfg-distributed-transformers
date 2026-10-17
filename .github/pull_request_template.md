## Qué cambia y por qué

<!-- Resumen breve. Enlaza la fase o la casilla de docs/roadmap.md a la que corresponde. -->

## Cómo se ha verificado

<!-- Tests nuevos, comprobaciones manuales, runs en GPU, AppTest de la web... -->

## Checklist

- [ ] `uv run pytest -q` en verde y `uv run ruff check src scripts tests` sin avisos
- [ ] Los tests de caracterización (`test_behaviour_golden.py`) pasan sin cambiar valores, o el cambio está justificado aquí
- [ ] Si arregla un fallo, hay un test que falla sin el arreglo
- [ ] `CHANGELOG.md` actualizado ([Unreleased])
- [ ] `docs/roadmap.md` actualizado (casillas y "Estado actual")
- [ ] Documentación afectada al día (`AGENTS.md`, `docs/`)
- [ ] Sin `Co-Authored-By` ni "Generated with" en commits ni en este texto
