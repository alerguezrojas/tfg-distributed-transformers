# Flujo de trabajo con IA

Este proyecto se desarrolla con asistentes de IA (Claude Code), con la misma disciplina que se
exige al código escrito a mano: revisión, tests, trazabilidad y nada que dependa de la memoria de
un chat. Este documento describe el "arnés" que lo hace posible; todo él está versionado en el
repositorio, así que se aplica igual en cualquier máquina y en cualquier sesión.

## Piezas

| Pieza | Dónde | Qué hace |
|---|---|---|
| Instrucciones para agentes | `AGENTS.md` (estándar abierto) y `CLAUDE.md` | Comandos, mapa del código, invariantes de diseño y convenciones. `CLAUDE.md` importa `AGENTS.md` y añade lo propio de Claude Code |
| Estado del proyecto | `docs/roadmap.md` → "Estado actual" | Hecho, en curso, siguiente paso y decisiones pendientes. Es lo primero que se lee al retomar |
| Agentes especializados | `.claude/agents/` | revisor (opus), implementador, documentador e investigador (sonnet), explorador (haiku) |
| Skills (procedimientos) | `.claude/skills/` | `/estado`, `/cerrar-bloque`, `/revisar-pr`, `/nuevo-dataset` |
| Hooks | `.claude/settings.json` + `.claude/hooks/` | Al abrir sesión inyecta el "Estado actual"; al editar Python pasa ruff; bloquea commits con atribución de IA |
| Permisos | `.claude/settings.json` | Permite sin preguntar los comandos de solo lectura y de test; prohíbe `push --force`, `reset --hard` y push directo a `main` |
| Lint y pre-commit | `pyproject.toml` (`[tool.ruff]`), `.pre-commit-config.yaml`, CI | Los mismos controles en local, en los hooks y en GitHub Actions |
| Plantilla de PR | `.github/pull_request_template.md` | Checklist de verificación, documentación y convenciones |
| Calidad continua | SonarCloud, CI | Cada PR pasa tests, lint y la puerta de calidad antes de mergear |

## Cómo se reparte el trabajo

- **La sesión principal coordina**: decide el diseño, delega, revisa e integra. Los cambios en el
  núcleo de entrenamiento, el predictor o la metodología los hace ella o pasan por el `revisor`.
- **Los agentes trabajan en paralelo y aislados** (`isolation: worktree`), cada uno en su rama y
  con su PR a `develop`. Ninguno mergea: integra la sesión principal tras revisar.
- **Cada tarea usa el modelo que necesita**: opus para revisar y diseñar, sonnet para implementar,
  documentar e investigar, haiku para buscar. Así se gasta menos sin perder calidad donde importa.

## Garantías de corrección

- **Tests de caracterización** (`tests/integration/test_behaviour_golden.py`): fijan los números
  exactos del entrenamiento. Un refactor no puede cambiarlos sin que se note.
- **Verificación en GPU** de los cambios del núcleo: el código anterior y el nuevo se ejecutan con
  la misma semilla y se comparan los CSV byte a byte.
- **Revisión independiente** antes de mergear lo importante, y SonarCloud en cada PR.
- **Resultados con semillas**: ninguna cifra se compara con la literatura sin repeticiones.

## Para quien retome el proyecto

1. Leer `docs/roadmap.md` → "Estado actual" (Claude Code lo muestra solo al abrir sesión).
2. `uv sync` y, una vez por clon, `uvx pre-commit install --hook-type pre-commit --hook-type commit-msg`.
3. Con Claude Code: `/estado` para el resumen y `/cerrar-bloque` al terminar cada bloque.
