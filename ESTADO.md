# Estado de VERODE

**Diagnóstico: EN_MARCHA** — actualizado el 26/11/2026 a las 23:52 (hora de Canarias).

Se actualiza cada 20 minutos. Si esta hora tiene más de una hora, el vigilante o el
nodo de login de VERODE han caído. Diagnósticos: EN_MARCHA, EN_MARCHA_CON_FALLOS (algún
run falló y la cola siguió), BLOQUEO (hay trabajo pero nada escribe desde hace
120 min), PARADO (no hay trabajo y la cola no ha terminado), TERMINADO.

## Trabajos en Slurm
```
13305 v100p36 RUNNING 3:18:43 verode21
```
## v100_panda36_26112026.out
```
=== swin36_s42 (configs/train_panda_36.yaml, semilla 42) 26/11 20:33
=== swin16_s42 (configs/train_panda.yaml, semilla 42) 26/11 21:42
=== swin36_s43 (configs/train_panda_36.yaml, semilla 43) 26/11 22:23
=== swin36_s44 (configs/train_panda_36.yaml, semilla 44) 26/11 23:28
```
## queue_next_26112026.out
```
(aún no ha empezado)
```
## Último entrenamiento
`/home/bejeque/alu0101317038/tfg-distributed-transformers/logs/verode/panda/single/swin_tiny_patch4_window7_224/epoch_metrics_26112026_232857.csv` (modificado: 26/11 23:51)
```
época, s/época, QWK val
3, 212.07, 0.760493
4, 212.57, 0.797662
5, 211.98, 0.800214
```
## QWK de test por run (eval normal y TTA x8)
```
swin16_s42         eval_test.out          0.8206
swin36_s42         eval_test.out          0.8485
swin36_s43         eval_test.out          0.8288
```
