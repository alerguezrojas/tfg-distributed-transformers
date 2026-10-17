# BigEarthNet-S2 v2.0

> Dataset original del TFG. Rutas, descarga, extracción y estructura.

## En local (SSD)
- **Dataset:** `/media/alejandro/SSD/datasets/bigearthnet/BigEarthNet-S2/`
- **Metadata:** `/media/alejandro/SSD/datasets/bigearthnet/metadata.parquet`

## En el clúster VERODE
- **Dataset:** `~/datasets/bigearthnet/BigEarthNet-S2/` ✓ completo (549 488 patches verificados)
- **Metadata:** `~/datasets/bigearthnet/metadata.parquet` ✓
- **Archivo comprimido:** `~/datasets/bigearthnet/BigEarthNet-S2.tar.zst` ✓ guardado (63 GB)

### Descarga (Zenodo record 10891137)
```bash
nohup wget -c "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1" \
     -O ~/datasets/bigearthnet/BigEarthNet-S2.tar.zst >> ~/logs/download.log 2>&1 &

wget -c "https://zenodo.org/records/10891137/files/metadata.parquet?download=1" \
     -O ~/datasets/bigearthnet/metadata.parquet
```

### Extracción
```bash
# OJO: usar ruta absoluta a zstd — el PATH del login node no incluye conda
nohup tar --use-compress-program=/home/bejeque/alu0101317038/miniconda3/bin/zstd \
    -xf ~/datasets/bigearthnet/BigEarthNet-S2.tar.zst \
    -C ~/datasets/bigearthnet/ >> ~/logs/extract.log 2>&1 &
```

### Verificar dataset
```bash
find ~/datasets/bigearthnet/BigEarthNet-S2/ -mindepth 2 -maxdepth 2 -type d | wc -l
# Debe dar 549 488
```

## Estructura y descripción
- **Directorios:** `root/scene_id/patch_id/*.tif` — `scene_id` = `patch_id` sin los dos últimos segmentos
- **Splits:** Train 237 871 | Val 122 342 | Test 119 825
- **Bandas:** B04, B03, B02 (proxy RGB), reflectancia / 10 000, clipped [0, 1]
- **Tarea:** clasificación multi-label, 19 clases CORINE Land Cover
- **Pérdida:** `BCEWithLogitsLoss` por defecto (sin sigmoid en el modelo); opcionalmente `FocalLoss` o `pos_weight` vía `training.loss`/`pos_weight` (ver `src/training/losses.py`)
- **Métricas:** macro F1 + sample-averaged accuracy + precision + recall
