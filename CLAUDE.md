@AGENTS.md

## Notas específicas para Claude Code

- Las instrucciones compartidas están en `AGENTS.md` (importado arriba). Este fichero solo añade lo
  propio de Claude Code; mantenerlo corto.
- Al autor le sirve ir paso a paso: un paso cada vez para tareas operativas (clúster, entrenamientos),
  explicaciones llanas y una recomendación clara cuando delega una decisión técnica.
- El historial detallado hasta la entrega está en [`docs/historial_tfg.md`](docs/historial_tfg.md):
  se consulta, no se edita.

## Trabajo con agentes y elección de modelo

La sesión principal coordina: decide, integra y revisa. Delega el trabajo independiente en los
agentes de proyecto (`.claude/agents/`), cada uno con el modelo que su tarea necesita:

| Agente | Modelo | Para qué |
|---|---|---|
| `explorador` | haiku | Localizar código, logs o runs (solo lectura, barato) |
| `implementador` | sonnet | Tareas acotadas: feature pequeña, arreglo, tests, refactor local |
| `documentador` | sonnet | Sincronizar documentación, roadmap y CHANGELOG con el código |
| `investigador` | sonnet | Bibliografía y referencias verificadas (`docs/research/`) |
| `revisor` | opus | Revisión independiente antes de mergear cambios del núcleo |

- Diseño, decisiones de metodología y cambios en el núcleo de entrenamiento o en el predictor:
  sesión principal (o `revisor`).
- Agentes en paralelo con `isolation: worktree` cuando hay un entrenamiento u otro trabajo en el
  directorio principal; cada uno abre su PR a `develop` y la sesión principal lo revisa y mergea.
- Las copias de los agentes van a `.claude/worktrees/` (ignorado por git).

## Continuidad entre sesiones

El proyecto debe poder retomarse desde cualquier máquina leyendo solo el repositorio. Al cerrar
cada bloque de trabajo (un PR mergeado, un experimento terminado, una decisión tomada), sin esperar
a que lo pidan:

1. Actualizar la sección **"Estado actual"** de [`docs/roadmap.md`](docs/roadmap.md): hecho, en
   curso, siguiente paso y decisiones pendientes del autor o del tutor.
2. Marcar las casillas del roadmap, añadir la entrada en `CHANGELOG.md` y actualizar la
   documentación afectada (`AGENTS.md`, `docs/`).
3. La memoria automática de Claude Code (fuera del repo, solo en esta máquina) guarda únicamente
   preferencias personales; nada que haga falta para continuar el proyecto puede vivir solo ahí.
