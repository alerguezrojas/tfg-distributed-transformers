# Modelos

ParaViT-Lab no está atado a timm: el modelo se elige con `model.name` y se resuelve así
(`src/models/backbones.py`):

| `model.name` | Qué se construye |
|---|---|
| un nombre del **registro** (p. ej. `ijepa_vith14_1k`) | el `BackboneSpec` registrado con ese nombre |
| `hf:<repositorio>` (p. ej. `hf:facebook/dinov2-small`) | adaptador genérico de Hugging Face `transformers` (`AutoModel`) |
| cualquier otro (p. ej. `vit_base_patch16_224`) | un modelo de **timm**, igual que siempre |

Los configs que ya existen no cambian: todo lo que no está en el registro ni empieza por `hf:`
sigue siendo timm, con el mismo código, los mismos pesos y el mismo comportamiento.

| Nombre | Origen | Pooling | Licencia | Notas |
|---|---|---|---|---|
| cualquier ID de timm | timm | el del modelo | la del modelo | ViT, DeiT, Swin, ResNet, ConvNeXt… |
| `ijepa_vith14_1k` | `transformers` (`facebook/ijepa_vith14_1k`) | media de tokens | **CC BY-NC 4.0** | ViT-H/14, 224 px, ImageNet-1k |
| `ijepa_vith14_22k` | `transformers` (`facebook/ijepa_vith14_22k`) | media de tokens | **CC BY-NC 4.0** | ViT-H/14, 224 px, ImageNet-22k |
| `hf:<repo>` | `transformers` | media de tokens (`model.pooling` lo cambia) | la del repositorio | cualquier ViT con `embeddings` y una lista de bloques (`encoder.layer`, `encoder.layers` o `layers`) |

## Cómo se usa cada tipo

**timm** (sin cambios): `model: {name: vit_small_patch16_224, pretrained: true}`.

**Backbone no timm**: requiere la dependencia opcional, que no se instala por defecto:

```bash
uv sync --extra hf        # o: pip install 'tfg-distributed-transformers[hf]'
```

```yaml
model:
  name: ijepa_vith14_1k      # o hf:facebook/ijepa_vith14_1k
  pretrained: true
  freeze_backbone: true      # sonda lineal (ver abajo)
  # pooling: mean            # 'mean' (por defecto) o 'cls'
```

Si falta `transformers`, el error lo dice y da el comando de instalación. Los tests no la
necesitan: usan un módulo falso.

## Cómo añadir un modelo

1. **Escribir el codificador**: un `nn.Module` que reciba `(B, C, H, W)` y devuelva `(B, D)`, con
   el atributo `num_features = D`. Si quieren LLRD, que ofrezca `llrd_layout()` y devuelva
   `(parámetros del embedding, [parámetros de cada bloque], resto)` como listas de
   `(nombre, parámetro)` (ejemplo: `HFEncoder`).
2. **Registrar la ficha** en un módulo que se importe (p. ej. al final de `src/models/backbones.py`):

   ```python
   from src.models.backbones import BackboneSpec, register_backbone

   register_backbone(BackboneSpec(
       name="mi_modelo",
       source="transformers",            # 'timm' | 'transformers' | 'torch_hub'
       build=lambda cfg: MiEncoder(...), # cfg: pretrained, in_chans, img_size, pooling
       description="Qué es",
       pooling="mean", patch_size=16, default_img_size=224,
       mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
       llrd=False, license="Apache-2.0", non_commercial=False,
   ))
   ```

3. **Usarlo** con `model.name: mi_modelo` en el config. No hace falta tocar el entrenamiento,
   las estrategias ni los decoradores. (Para un repositorio de Hugging Face sin código propio
   basta `hf:<repo>`, sin registrar nada.)
4. **Probarlo** sin red: `tests/unit/test_backbones.py` muestra cómo simular el modelo con un
   módulo falso.

## Sonda lineal frente a ajuste fino

- **Ajuste fino** (por defecto): se entrena todo el modelo; con `training.llrd_decay > 0` las capas
  más profundas aprenden más despacio (LLRD).
- **Sonda lineal** (`model.freeze_backbone: true`): el codificador queda congelado (y en modo
  evaluación) y solo se entrena la cabeza `Dropout → Linear`. El optimizador recibe solo los
  parámetros con `requires_grad`, y `llrd_decay` no se aplica (el log lo avisa). Mide la calidad
  de las representaciones sin que el ajuste las tape, y es mucho más barata.

Para modelos de tipo JEPA, preentrenados sin etiquetas, la sonda lineal es la comparación
natural; el ajuste fino con LLRD es la segunda.

## Normalización de la entrada

Los transforms del dataset normalizan con la media y la desviación de ImageNet (RGB). Un
backbone que declare otras (`BackboneSpec.mean/std`) **no las recibe**: el log avisa de la
diferencia al construirlo. I-JEPA usa las de ImageNet, así que encaja. Si un modelo nuevo
necesita otras, hoy hay que ajustar los transforms del dataset; queda como mejora pendiente
para no tocar los datasets por un caso que todavía no existe.

## Limitaciones

- **Paralelismo de modelo**: solo para ViT de timm. Con un backbone registrado o `hf:` el validador
  da un error claro.
- **Canales**: los backbones de `transformers` esperan 3 canales (no sirven para datasets en
  escala de grises).
- **Resolución**: si la entrada no mide lo que el modelo espera, se interpolan los embeddings de
  posición (`interpolate_pos_encoding`); depende de que el modelo lo admita.
- **Predictor** (`paravit estimate`, `src/model_profile.py`): solo perfila modelos de timm. Para
  I-JEPA ViT-H/14 se puede estimar con el equivalente de timm `vit_huge_patch14_224`.

## Licencias

Todos los pesos de I-JEPA son **CC BY-NC 4.0**: no se pueden usar con fines comerciales. Al
construir un modelo marcado como no comercial (`non_commercial=True`) el log lo avisa una vez.
Antes de registrar un modelo nuevo, comprueben la licencia de los **pesos** (a menudo distinta
de la del código) y anótenla en la ficha. Para los datasets, ver [datasets/](../datasets/README.md).

## Protocolo de comparación recomendado

Para que dos modelos se puedan comparar de forma justa (por ejemplo, JEPA frente a un ViT
supervisado en PANDA):

1. **Mismas particiones** (el `splits.csv` fijo) y mismos transforms.
2. **Tres semillas** por modelo y configuración; se compara la media ± desviación, no una sola
   ejecución (con una semilla, diferencias de ±0,02 de QWK son ruido).
3. **Métricas**: QWK en test como principal y los cortes clínicos ISUP ≥ 1 y ISUP ≥ 2
   (sensibilidad, especificidad y AUC, con `eval --save-predictions`).
4. **Sonda lineal y ajuste fino** de cada modelo, a igual resolución y número de épocas.
5. Anotar la **licencia** y el origen de los pesos junto a cada resultado.
