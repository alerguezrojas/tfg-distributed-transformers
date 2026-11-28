# Estado de VERODE

**Diagnóstico: EN_MARCHA** — actualizado el 28/11/2026 a las 00:28 (hora de Canarias).

Se actualiza cada 20 minutos. Si esta hora tiene más de una hora, el vigilante o el
nodo de login de VERODE han caído. Diagnósticos: EN_MARCHA, EN_MARCHA_CON_FALLOS (algún
run falló y la cola siguió), BLOQUEO (hay trabajo pero nada escribe desde hace
120 min), PARADO (no hay trabajo y la cola no ha terminado), TERMINADO.

## Trabajos en Slurm
```
13307 v100conf RUNNING 1:21:04 verode21
```
## queue_confirm_27112026.out
```
=== vitb36_s43 (configs/train_panda_36.yaml, semilla 43, ) 27/11 23:07
```
## Último entrenamiento
`/home/bejeque/alu0101317038/tfg-distributed-transformers/logs/verode/panda/single/vit_base_patch16_224.augreg2_in21k_ft_in1k/epoch_metrics_27112026_230744.csv` (modificado: 28/11 00:22)
```
época, s/época, QWK val
5, 475.37, 0.77682
6, 476.51, 0.793914
7, 476.82, 0.81255
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
```
