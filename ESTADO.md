# Estado de VERODE

**Diagnóstico: EN_MARCHA** — actualizado el 26/11/2026 a las 22:12 (hora de Canarias).

Se actualiza cada 20 minutos. Si esta hora tiene más de una hora, el vigilante o el
nodo de login de VERODE han caído. Diagnósticos: EN_MARCHA, EN_MARCHA_CON_FALLOS (algún
run falló y la cola siguió), BLOQUEO (hay trabajo pero nada escribe desde hace
120 min), PARADO (no hay trabajo y la cola no ha terminado), TERMINADO.

## Trabajos en Slurm
```
13305 v100p36 RUNNING 1:38:26 verode21
```
## v100_panda36_26112026.out
```
=== swin36_s42 (configs/train_panda_36.yaml, semilla 42) 26/11 20:33
=== swin16_s42 (configs/train_panda.yaml, semilla 42) 26/11 21:42
```
## queue_next_26112026.out
```
(aún no ha empezado)
```
## Último entrenamiento
`/home/bejeque/alu0101317038/tfg-distributed-transformers/logs/verode/panda/single/swin_tiny_patch4_window7_224/epoch_metrics_26112026_214254.csv` (modificado: 26/11 22:11)
```
época, s/época, QWK val
9, 106.73, 0.809712
10, 106.01, 0.824192
11, 105.52, 0.814499
```
## QWK de test por run (eval normal y TTA x8)
```
swin36_s42         eval_test.out          0.8485
```
