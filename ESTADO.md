# Estado de VERODE

**Diagnóstico: EN_MARCHA** — actualizado el 27/11/2026 a las 03:33 (hora de Canarias).

Se actualiza cada 20 minutos. Si esta hora tiene más de una hora, el vigilante o el
nodo de login de VERODE han caído. Diagnósticos: EN_MARCHA, EN_MARCHA_CON_FALLOS (algún
run falló y la cola siguió), BLOQUEO (hay trabajo pero nada escribe desde hace
120 min), PARADO (no hay trabajo y la cola no ha terminado), TERMINADO.

## Trabajos en Slurm
```
13306 v100cola RUNNING 3:14:24 verode21
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
=== TTA de los runs de la tarde 27/11 00:19
swin16_s42 tta8:   QWK (cuadrático)  : 0.8314
swin36_s42 tta8:   QWK (cuadrático)  : 0.8519
swin36_s43 tta8:   QWK (cuadrático)  : 0.8393
swin36_s44 tta8:   QWK (cuadrático)  : 0.8082
=== swin36_1024_s42 (configs/train_panda_36.yaml, semilla 42, --set data.image_size=1024 --set model.img_size=1024) 27/11 00:51
swin36_1024_s42 tta1:   QWK (cuadrático)  : 0.8444
swin36_1024_s42 tta8:   QWK (cuadrático)  : 0.8518
=== swin64_s42 (configs/train_panda_36.yaml, semilla 42, --set data.mosaics=mosaics_l1_64x192) 27/11 02:55
```
## Último entrenamiento
`/home/bejeque/alu0101317038/paravit_wt/logs/verode/panda/single/swin_tiny_patch4_window7_224/epoch_metrics_27112026_025531.csv` (modificado: 27/11 03:31)
```
época, s/época, QWK val
6, 212.31, 0.79477
7, 211.83, 0.823197
8, 211.39, 0.813252
```
## QWK de test por run (eval normal y TTA x8)
```
swin16_s42         eval_test.out          0.8206
swin16_s42         eval_test_tta8.out     0.8314
swin36_1024_s42    eval_test.out          0.8444
swin36_1024_s42    eval_test_tta8.out     0.8518
swin36_s42         eval_test.out          0.8485
swin36_s42         eval_test_tta8.out     0.8519
swin36_s43         eval_test.out          0.8288
swin36_s43         eval_test_tta8.out     0.8393
swin36_s44         eval_test.out          0.7998
swin36_s44         eval_test_tta8.out     0.8082
```
