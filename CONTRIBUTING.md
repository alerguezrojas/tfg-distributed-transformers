# Cómo colaborar

Guía corta de cómo se trabaja en ParaViT-Lab. Si es su primera vez en el proyecto, empiecen por
[docs/primeros_pasos.md](docs/primeros_pasos.md).

## Ramas

```
main  ←  develop  ←  feature/*   (y ramas largas de colaborador: tfg/<nombre>)
```

- `main` y `develop` están protegidas: solo se cambian con un pull request que pase el check `tests`
  de la CI. No se hace push directo a ninguna de las dos.
- Cada cambio sale de `develop` en una rama `feature/<tema>` (o `fix/`, `docs/`, `chore/`) y vuelve
  a `develop` con un PR. GitHub borra la rama al mergear.
- `develop` pasa a `main` solo cuando el autor lo valida, como versión etiquetada (`v1.1.0`, etc.)
  con su entrada en el [CHANGELOG](CHANGELOG.md).

### Rama larga de un colaborador

Quien lleve un trabajo extenso (por ejemplo, un TFG con otros modelos) puede tener una rama propia,
`tfg/<nombre>`, que sale de `develop`:

```bash
git switch develop && git pull
git switch -c tfg/<nombre>
git push -u origin tfg/<nombre>
```

- Se mantiene al día con `git merge origin/develop` (no rebase: la rama es compartida y su
  historial no se reescribe).
- Lo que sea útil para todos (un backbone, un dataset, una corrección) se aporta a `develop` con
  **PR pequeños** desde una rama `feature/*` que salga de `develop`, no con un PR gigante al final.
- Los experimentos exploratorios se quedan en la rama larga; a `develop` solo llega lo que tiene
  tests y documentación.

## Commits

[Conventional Commits](https://www.conventionalcommits.org/es/) en español, cortos: `feat:`,
`fix(web):`, `docs:`, `chore:`, `test:`. Sin `Co-Authored-By` ni «Generated with…» en commits ni en
PR (un hook lo comprueba).

## Definición de hecho

Un cambio está listo para revisión cuando:

1. `uv run pytest -q` está en verde y `uv run ruff check src scripts tests` no da avisos
   (los tests son solo de CPU, sin GPU ni datos).
2. Hay tests para la lógica nueva; si arregla un fallo, hay un test que falla sin el arreglo.
3. `tests/integration/test_behaviour_golden.py` pasa sin tocar sus valores: si falla, el
   comportamiento ha cambiado y se investiga.
4. La documentación afectada está al día (`docs/`, `AGENTS.md`, fichas de dataset).
5. `CHANGELOG.md` tiene su entrada en *Unreleased*, y si cierra una casilla del roadmap, esta se marca
   en [docs/roadmap.md](docs/roadmap.md).

Las invariantes de diseño (el bucle vive solo en `EpochController.fit()`, el núcleo no asume un
dataset, el predictor es puro, etc.) están en [AGENTS.md](AGENTS.md); léanlas antes de tocar el
entrenamiento.

## pre-commit

Una vez por clon:

```bash
uvx pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Antes de cada commit pasan ruff, la comprobación de YAML, ficheros grandes (más de 2 MB) y
**gitleaks**, que rechaza tokens de Kaggle, Hugging Face y claves en el diff. Nunca suban
credenciales, datos médicos ni checkpoints; las reglas completas están en [SECURITY.md](SECURITY.md).

## Pedir revisión

Abran el PR contra `develop` con la plantilla (qué cambia, cómo se ha verificado, lista de
comprobación) y asignen como revisor al autor. La CI ejecuta tests, lint, la auditoría de
dependencias y CodeQL, y SonarCloud analiza el resultado. Si el cambio toca el entrenamiento, el
predictor o los datos, indiquen en el PR cómo se ha comprobado que el comportamiento no cambia.
El merge lo hace el autor, no quien abre el PR.

## Incidencias

Para un fallo o una propuesta, usen las plantillas de *Issues*. Las vulnerabilidades **no** van
en una incidencia pública: se informan por el aviso privado de seguridad ([SECURITY.md](SECURITY.md)).

## Más documentación

[Primeros pasos](docs/primeros_pasos.md) · [AGENTS.md](AGENTS.md) ·
[arquitectura](docs/arquitectura.md) · [guía de uso](docs/guia_de_uso.md) ·
[datasets](docs/datasets/README.md) · [modelos](docs/modelos/README.md) ·
[roadmap y estado actual](docs/roadmap.md) · [Kaggle](kaggle/README.md)
