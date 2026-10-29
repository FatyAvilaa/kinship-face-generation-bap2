# Data

This repository **does not include any face images or face embeddings**. It only contains the metadata
needed to rebuild exactly the same splits.

## What is included

| File | Contents |
|---|---|
| `splits/train_families.txt`, `val_families.txt`, `test_families.txt` | Family IDs in each split (630 / 79 / 79). |
| `splits/train_identity.csv`, `val_identity.csv`, `test_identity.csv` | One row per unique father–mother–child triad (1,571 / 196 / 212). Used for training and evaluation. |
| `splits/train_sampled_k20.csv` | Up to 20 photo combinations per training triad (28,666 rows). |
| `splits/all_unique_images.csv` | Every image referenced by the splits (21,346 rows, column `ruta_imagen`). |
| `ages/mivolo_ages.csv` | Age estimated with MiVOLO v2 for the 1,967 children (`key`, `age`). |
| `gender_dict.pkl` | Python dict `{"F0001/MID1": "man" \| "woman", ...}` used to build the prompts. |

### Triad CSV columns

The column names were kept as they were produced by notebook 01 (in Spanish), because the code and the
notebooks read them directly:

| Column | Meaning |
|---|---|
| `p1_papa` | Father's photo, relative to `FIDs/` (e.g. `F0003/MID2/P00019_face2.jpg`) |
| `p1_mama` | Mother's photo |
| `p2` | Child's photo |
| `carpeta_papa`, `carpeta_mama`, `carpeta_hijo` | Person folder of the father, mother and child (`FXXXX/MIDY`) |
| `familia`, `familia_papa`, `familia_mama` | Family ID (`FXXXX`) |

## What is missing and how to get it

1. **FIW images.** Request the *Families in the Wild* dataset at
   <https://web.northeastern.edu/smilelab/fiw/> (research use only) and place the `FIDs/` folder so that
   paths look like this:

   ```
   data/raw/FIDs/F0001/MID1/P00001_face2.jpg
   ```

2. **ArcFace embeddings** (`data/embeddings/arcface_insightface.pkl`). Generate them with the
   *"GPU extraction"* cell in [`notebooks/02_arcface_validation.ipynb`](../notebooks/02_arcface_validation.ipynb)
   (InsightFace `buffalo_l`, 320×320 detection). The result is a dict
   `{"F0001/MID1/P00001_face2.jpg": np.ndarray(512,)}` with about 26.5 k entries.

3. *(Optional)* **Expanded triads** (`data/processed/tripletes_fiw.csv`), produced by
   [`notebooks/01_database_exploration.ipynb`](../notebooks/01_database_exploration.ipynb). Only needed if you
   want to rebuild the splits from scratch; the splits above are already included.

When everything is in place, `data/` should look like this:

```
data/
├── raw/FIDs/...                         # FIW images (not versioned)
├── embeddings/arcface_insightface.pkl   # ArcFace embeddings (not versioned)
├── splits/                              # included
├── ages/mivolo_ages.csv                 # included
└── gender_dict.pkl                      # included
```

The paths the code expects are defined in [`src/config.py`](../src/config.py).

## Terms of use

FIW is distributed by its authors for **non-commercial research only**. The split files above only contain
relative file paths and family IDs, not images. To use them you must obtain FIW from its authors and follow
their terms. See [`THIRD_PARTY_LICENSES.md`](../THIRD_PARTY_LICENSES.md).
