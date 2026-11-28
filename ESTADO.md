# Estado de VERODE

**Diagnóstico: EN_MARCHA** — actualizado el 28/11/2026 a las 13:11 (hora de Canarias).

Se actualiza cada 20 minutos. Si esta hora tiene más de una hora, el vigilante o el
nodo de login de VERODE han caído. Diagnósticos: EN_MARCHA, EN_MARCHA_CON_FALLOS (algún
run falló y la cola siguió), BLOQUEO (hay trabajo pero nada escribe desde hace
120 min), PARADO (no hay trabajo y la cola no ha terminado), TERMINADO.

## Trabajos en Slurm
```
13307 v100conf RUNNING 14:03:42 verode21
```
## queue_confirm_27112026.out
```
=== vitb36_s43 (configs/train_panda_36.yaml, semilla 43, ) 27/11 23:07
vitb36_s43 tta1:   QWK (cuadrático)  : 0.8506
vitb36_s43 tta8:   QWK (cuadrático)  : 0.8545
=== vitb36_s44 (configs/train_panda_36.yaml, semilla 44, ) 28/11 02:10
vitb36_s44 tta1:   QWK (cuadrático)  : 0.8594
vitb36_s44 tta8:   QWK (cuadrático)  : 0.8506
=== swinv2_36_s43 (configs/train_panda_36.yaml, semilla 43, ) 28/11 05:13
swinv2_36_s43 tta1:   QWK (cuadrático)  : 0.8515
swinv2_36_s43 tta8:   QWK (cuadrático)  : 0.8555
=== swinv2_36_s44 (configs/train_panda_36.yaml, semilla 44, ) 28/11 06:40
swinv2_36_s44 tta1:   QWK (cuadrático)  : 0.8331
swinv2_36_s44 tta8:   QWK (cuadrático)  : 0.8436
=== vitb36_1024_s42 (configs/train_panda_36.yaml, semilla 42, --set data.image_size=1024 --set model.img_size=1024) 28/11 08:05
```
## Último entrenamiento
`/home/bejeque/alu0101317038/tfg-distributed-transformers/logs/verode/panda/single/vit_base_patch16_224.augreg2_in21k_ft_in1k/epoch_metrics_28112026_080600.csv` (modificado: 28/11 12:54)
```
época, s/época, QWK val
11, 1179.56, 0.841095
12, 1174.47, 0.828094
13, 1169.8, 0.849806
```
## QWK de test por run (eval normal y TTA x8)
```
swin16_s42         eval_test.out          0.8206
swin16_s42         eval_test_tta8.out     0.8314
swin36_1024_s42    eval_test.out          0.8444
swin36_1024_s42    eval_test_tta8.out     0.8518
swin36_1024_s43    eval_test.out          0.8244
swin36_1024_s43    eval_test_tta8.out     0.8384
swin36_1024_s44    eval_test.out          0.8411
swin36_1024_s44    eval_test_tta8.out     0.8451
swin36_s42         eval_test.out          0.8485
swin36_s42         eval_test_tta8.out     0.8519
swin36_s43         eval_test.out          0.8288
swin36_s43         eval_test_tta8.out     0.8393
swin36_s44         eval_test.out          0.7998
swin36_s44         eval_test_tta8.out     0.8082
swin64_s42         eval_test.out          0.8293
swin64_s42         eval_test_tta8.out     0.8383
swin64_s43         eval_test.out          0.8262
swin64_s43         eval_test_tta8.out     0.8269
swin64_s44         eval_test.out          0.8142
swin64_s44         eval_test_tta8.out     0.8172
swinv2_36_s42      eval_test.out          0.8475
swinv2_36_s42      eval_test_tta8.out     0.8491
swinv2_36_s43      eval_test.out          0.8515
swinv2_36_s43      eval_test_tta8.out     0.8555
swinv2_36_s44      eval_test.out          0.8331
swinv2_36_s44      eval_test_tta8.out     0.8436
vitb36_s42         eval_test.out          0.8409
vitb36_s42         eval_test_tta8.out     0.8508
vitb36_s43         eval_test.out          0.8506
vitb36_s43         eval_test_tta8.out     0.8545
vitb36_s44         eval_test.out          0.8594
vitb36_s44         eval_test_tta8.out     0.8506
```
