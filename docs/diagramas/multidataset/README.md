# Diagramas del diseño multi-dataset

Fuentes PlantUML de los diagramas que explican el diseño para varios datasets (ficha
`DatasetSpec`, registro y estrategia `Task`): notación, patrones genéricos, el ejemplo del
corrector de exámenes, el antes (TFG) y el después, y la secuencia de un entrenamiento. La
explicación completa está en [datasets/README.md](../../datasets/README.md) y en
[arquitectura.md](../../arquitectura.md#datasets-y-tareas).

Para regenerarlos hace falta Java y el `.jar` de PlantUML; el *layout* `smetana` evita depender
de Graphviz:

```bash
java -jar plantuml.jar -tsvg docs/diagramas/multidataset/*.puml
```
