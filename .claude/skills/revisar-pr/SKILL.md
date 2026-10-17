---
name: revisar-pr
description: Revisa un PR antes de mergearlo — lanza el agente revisor sobre su diff, comprueba CI y SonarCloud y resume si se puede mergear. Úsala con el número del PR (por ejemplo, /revisar-pr 152).
---

Argumento: número del PR (`$ARGUMENTS`).

1. `gh pr view $ARGUMENTS --json title,headRefName,baseRefName,body,files` y `gh pr checks $ARGUMENTS`.
2. Lanza el agente **revisor** (modelo opus, solo lectura) sobre `git diff develop...<rama del PR>`.
   Si el PR solo toca documentación o la web sin lógica, basta una revisión propia.
3. Si toca el núcleo de entrenamiento, el predictor o los datos, ejecuta también la suite completa
   sobre la rama del PR.
4. Resume: hallazgos del revisor (bug / riesgo / detalle), estado de CI y SonarCloud y una
   recomendación clara (mergear, mergear tras arreglos, no mergear). No mergees sin indicarlo.
