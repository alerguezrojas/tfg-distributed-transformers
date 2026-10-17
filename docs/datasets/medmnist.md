# MedMNIST

Colección de datasets biomédicos estandarizados (MedMNIST v2/v3) para clasificación de
imágenes. Cada subconjunto viene ya dividido en train/val/test y en un único fichero `.npz`.

> Yang, J., Shi, R., Wei, D., Liu, Z., Zhao, L., Ke, B., Pfister, H., Ni, B. *MedMNIST v2: A
> large-scale lightweight benchmark for 2D and 3D biomedical image classification*. Scientific
> Data 10, 41 (2023). Web: <https://medmnist.com>. Datos: Zenodo, registro 10519652.

## Subconjuntos integrados

| `data.name` | Subconjunto | Tarea | Canales | Clases | Train / Val / Test | Licencia |
|---|---|---|---|---|---|---|
| `medmnist_breast` | BreastMNIST, ecografía de mama | binaria (maligno) | 1 | 1 | 546 / 78 / 156 | CC BY 4.0 |
| `medmnist_derma` | DermaMNIST, dermatoscopia (HAM10000) | multiclase | 3 | 7 | 7 007 / 1 003 / 2 005 | **CC BY-NC 4.0** |
| `medmnist_path` | PathMNIST, histología colorrectal | multiclase | 3 | 9 | 89 996 / 10 004 / 7 180 | CC BY 4.0 |
| `medmnist_chest` | ChestMNIST, radiografía de tórax (ChestX-ray14) | multi-etiqueta | 1 | 14 | 78 468 / 11 219 / 22 433 | CC BY 4.0 |

- En **BreastMNIST** la etiqueta original es 0 = maligno y 1 = normal/benigno. La clase positiva
  del modelo es **maligno**.
- **DermaMNIST** es solo para uso no comercial; hay que citar también su fuente (HAM10000).
- Cada subconjunto deriva de un dataset original que conviene citar si se publica: BreastMNIST
  (Al-Dhabyani et al., 2020), DermaMNIST (HAM10000, Tschandl et al., 2018), PathMNIST
  (NCT-CRC-HE-100K, Kather et al., 2019) y ChestMNIST (ChestX-ray14, Wang et al., 2017).

## Descarga

```bash
# Los cuatro subconjuntos a 28 px (~310 MB en total), con verificación MD5
uv run python scripts/download_medmnist.py --root /media/alejandro/SSD/datasets/medmnist

# Solo algunos, o a otra resolución (64, 128 o 224 px; los de 224 pesan mucho más)
uv run python scripts/download_medmnist.py --root <carpeta> --datasets medmnist_breast --size 224
```

El lector usa directamente los `.npz` oficiales (`<subset>.npz` para 28 px y
`<subset>_<tamaño>.npz` para el resto), así que no hace falta el paquete `medmnist`.

## Configuración

```yaml
data:
  name: medmnist_breast
  root: "/media/alejandro/SSD/datasets/medmnist"   # carpeta con los .npz
  size: 28                                          # 28 (por defecto) | 64 | 128 | 224
```

Hay un config local por subconjunto: `configs/medmnist_{breast,derma,path,chest}_local.yaml`.

**Preprocesado:** las imágenes se escalan a [0, 1], se reescalan a 224×224 (la entrada del ViT) y
se normalizan: estadísticas de ImageNet en RGB y media/desviación 0,5 en escala de grises. **No se
aplica data augmentation**, igual que en el protocolo de referencia de MedMNIST, para que los
resultados sean comparables con los publicados. Con un solo canal, timm adapta la primera capa
de los pesos preentrenados.

Los artefactos de cada run van a `logs/{env}/medmnist_<subset>/{mode}/{model}/`.

## Referencias oficiales (test)

Tabla de [medmnist.com](https://medmnist.com), consultada el 17/10/2026. ACC es accuracy y AUC es
el área bajo la curva ROC (media macro).

| Método | Breast AUC / ACC | Derma AUC / ACC | Path AUC / ACC | Chest AUC / ACC |
|---|---|---|---|---|
| ResNet-18 (28) | 0.901 / 0.863 | 0.917 / 0.735 | 0.983 / 0.907 | 0.768 / 0.947 |
| ResNet-18 (224) | 0.891 / 0.833 | 0.920 / 0.754 | 0.989 / 0.909 | 0.773 / 0.947 |
| ResNet-50 (28) | 0.857 / 0.812 | 0.913 / 0.735 | 0.990 / 0.911 | 0.769 / 0.947 |
| ResNet-50 (224) | 0.866 / 0.842 | 0.912 / 0.731 | 0.989 / 0.892 | 0.773 / 0.948 |
| Google AutoML Vision | 0.919 / 0.861 | 0.914 / 0.768 | 0.944 / 0.728 | 0.778 / 0.948 |

## Resultados propios

Primer baseline, 17/10/2026: **vit_tiny_patch16_224** preentrenado en ImageNet, imágenes de 28 px
reescaladas a 224, sin augmentation, AdamW con LLRD 0.75, lr 1e-4, batch 128, semilla 42,
early stopping con paciencia 5. RTX 3060 Ti, un solo entrenamiento por dataset. Se evalúa una
vez en **test** el checkpoint con mejor F1 de validación.

| Dataset | Épocas (mejor) | Test AUC | Test ACC | Rango oficial AUC / ACC | Tiempo/época | Energía total |
|---|---|---|---|---|---|---|
| BreastMNIST | 22 (17) | **0.889** | **0.840** | 0.857–0.919 / 0.812–0.863 | 1,4 s | 1,4 Wh |
| DermaMNIST | 15 (15) | **0.931** | **0.765** | 0.912–0.920 / 0.731–0.768 | 15 s | 11,5 Wh |
| PathMNIST | 5 (5) | **0.993** | **0.925** | 0.944–0.990 / 0.728–0.911 | 164 s | 44,4 Wh |
| ChestMNIST | 5 (5) | **0.772** | **0.948** | 0.768–0.778 / 0.947–0.948 | 142 s | 38,6 Wh |

Los rangos oficiales son el mínimo y el máximo de los métodos de la tabla anterior.

Lectura:
- **Los cuatro quedan dentro o por encima del rango oficial.** DermaMNIST y PathMNIST superan a
  todas las ResNet de referencia en AUC; BreastMNIST y ChestMNIST quedan dentro del rango.
- **No es una comparación rigurosa todavía:** un único run por dataset (sin repeticiones ni
  desviación) y otro modelo y protocolo de entrenamiento (ViT preentrenado en ImageNet frente a
  las ResNet y AutoML de la tabla oficial).
- **PathMNIST** baja de 0,987 de accuracy en validación a 0,925 en test: su test procede de otro
  centro clínico (desplazamiento de distribución conocido del dataset).
- **ChestMNIST** tiene un F1 a umbral 0,5 casi nulo (0,04): las 14 patologías son raras y el
  modelo casi nunca supera 0,5, así que la accuracy alta sale sobre todo de los negativos. Por eso
  la métrica de referencia es el AUC, y por eso elegir el checkpoint por F1@0,5 no es adecuado
  aquí (ver el roadmap).
- La **potencia media** medida (165–195 W) coincide con los 180 W calibrados para la RTX 3060 Ti
  con BigEarthNet: una validación independiente de esa constante del predictor.

Artefactos: `logs/local/medmnist_{breast,derma,path,chest}/single/vit_tiny_patch16_224/` (log,
CSVs por época, clase, batch y matriz de confusión, y `test_*.csv`).
