# Datos

Este repositorio **no incluye imágenes ni embeddings faciales**. Solo contiene los metadatos necesarios para
reproducir exactamente las mismas particiones.

## Qué está incluido

| Archivo | Contenido |
|---|---|
| `splits/train_families.txt`, `val_families.txt`, `test_families.txt` | Familias de cada partición (630 / 79 / 79). |
| `splits/train_identity.csv`, `val_identity.csv`, `test_identity.csv` | Una fila por tríada única padre–madre–hijo (1,571 / 196 / 212). Usadas para entrenar y evaluar. |
| `splits/train_sampled_k20.csv` | Hasta 20 combinaciones de fotos por tríada de entrenamiento (28,666 filas). |
| `splits/todas_imagenes_unicas.csv` | Lista de todas las imágenes referenciadas. |
| `ages/mivolo_ages.csv` | Edad estimada con MiVOLO v2 para los 1,967 hijos (`key`, `age`). |
| `gender_dict.pkl` | Diccionario `{"F0001/MID1": "man" | "woman", ...}` usado en los prompts. |

Columnas de los CSV de tríadas:

```
p1_papa, p1_mama, p2                         # ruta de la foto del padre, madre e hijo (relativa a FIDs/)
carpeta_papa, carpeta_mama, carpeta_hijo     # carpeta de cada persona (FXXXX/MIDY)
familia, familia_papa, familia_mama          # id de familia
```

## Qué falta y cómo obtenerlo

1. **Imágenes de FIW.** Solicitar el dataset *Families in the Wild* en
   <https://web.northeastern.edu/smilelab/fiw/> (uso para investigación) y colocar la carpeta `FIDs/` en:

   ```
   data/raw/FIDs/F0001/MID1/P00001_face2.jpg
   ```

2. **Embeddings ArcFace** (`data/embeddings/arcface_insightface.pkl`). Se generan con la celda
   *"Extracción con GPU"* de [`notebooks/02_arcface_validation.ipynb`](../notebooks/02_arcface_validation.ipynb)
   (InsightFace `buffalo_l`, detección a 320×320). El resultado es un diccionario
   `{"F0001/MID1/P00001_face2.jpg": np.ndarray(512,)}` con ~26.5 k entradas.

3. (Opcional) **Tríadas expandidas** (`data/processed/tripletes_fiw.csv`) con
   [`notebooks/01_database_exploration.ipynb`](../notebooks/01_database_exploration.ipynb).

Las rutas que espera el código están en [`src/config.py`](../src/config.py).
