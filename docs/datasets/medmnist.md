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

RESULTADOS_PENDIENTES
