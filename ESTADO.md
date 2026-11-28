# Estado de VERODE

**Diagnóstico: EN_MARCHA** — actualizado el 28/11/2026 a las 05:49 (hora de Canarias).

Se actualiza cada 20 minutos. Si esta hora tiene más de una hora, el vigilante o el
nodo de login de VERODE han caído. Diagnósticos: EN_MARCHA, EN_MARCHA_CON_FALLOS (algún
run falló y la cola siguió), BLOQUEO (hay trabajo pero nada escribe desde hace
120 min), PARADO (no hay trabajo y la cola no ha terminado), TERMINADO.

## Trabajos en Slurm
```
13307 v100conf RUNNING 6:42:10 verode21
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
```
## Último entrenamiento
`/home/bejeque/alu0101317038/tfg-distributed-transformers/logs/verode/panda/single/swinv2_tiny_window8_256/epoch_metrics_28112026_051404.csv` (modificado: 28/11 05:44)
```
época, s/época, QWK val
4, 251.44, 0.80036
5, 252.65, 0.810117
6, 252.55, 0.808952
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
vitb36_s42         eval_test.out          0.8409
vitb36_s42         eval_test_tta8.out     0.8508
vitb36_s43         eval_test.out          0.8506
vitb36_s43         eval_test_tta8.out     0.8545
vitb36_s44         eval_test.out          0.8594
vitb36_s44         eval_test_tta8.out     0.8506
```
