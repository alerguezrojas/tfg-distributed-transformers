---
name: estado
description: Resume dónde está el proyecto ahora mismo — sección "Estado actual" del roadmap, ramas y PRs abiertos, trabajo sin commitear y siguiente paso. Úsala al empezar una sesión o cuando el autor pregunte "¿por dónde vamos?".
---

1. Lee la sección **"Estado actual"** de `docs/roadmap.md` y las casillas abiertas de la fase en curso.
2. Comprueba el estado real del repositorio:
   - `git status --short` y `git branch -a`
   - `git log --oneline -5 develop` y `git log --oneline develop..main` / `main..develop`
   - `gh pr list --state open`
3. Señala cualquier discrepancia entre el roadmap y el repositorio (por ejemplo, un PR mergeado
   que el roadmap aún da como pendiente) y corrígela si es evidente.
4. Responde en español, breve: qué está hecho, qué está en curso, qué decide el autor y cuál es
   el siguiente paso recomendado.
