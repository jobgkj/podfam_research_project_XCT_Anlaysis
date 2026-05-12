# XCT Defect Segmentation and Evaluation
## for Additively Manufactured Metal Parts

---

## Project Overview

This repository implements a **reproducible, physics-informed workflow** for defect segmentation and evaluation in **industrial X-ray Computed Tomography (XCT)** data of additively manufactured metal components.

The pipeline combines **classical image processing-based segmentation** (global and local thresholding) with **deep learning-based defect segmentation**, enabling:

- Objective comparison of **global vs local thresholding methods**
- Generation of **weak pseudo-labels** for supervised learning
- Fair benchmarking between **2D and 3D U-Net architectures**
- Quantitative porosity and pore morphology analysis

Raw XCT data are **never committed** to the repository.

---

## Repository Structure

```
project_root/
├── config.py                        ← Central config (paths, hyperparameters)
├── pipeline.py                      ← Stage 5: ML training (2D + 3D U-Net)
├── evaluate_2d_vs_3d.py             ← Stage 6: Evaluation and visualization
│
├── scripts/
│   ├── run_preprocess.py            ← Stage 2: Preprocessing
│   └── compare_methods.py           ← Stage 3: Thresholding comparison
│
├── src/
│   ├── io.py                        ← I/O utilities (streaming TIFF loader)
│   ├── preprocess.py                ← Median + NLM filtering
│   ├── thresholding.py              ← Otsu, Yen, Bernsen implementations
│   └── metrics.py                   ← Slice-wise and volume metrics
│
├── data/
│   ├── loader.py                    ← Full-volume loader (used by pipeline.py)
│   ├── pseudo_labels.py             ← Pseudo-label generation
│   ├── dataset.py                   ← 2D patch dataset
│   ├── dataset_3d.py                ← 3D patch dataset
│   ├── raw/                         ← Raw TIFF stacks (not committed)
│   │   └── sample_01/
│   ├── processed/                   ← Preprocessed 8-bit slices (not committed)
│   │   └── sample_01/
│   └── masks/                       ← Pseudo-labels for training (not committed)
│       └── sample_01/
│
├── models/
│   ├── unet2d.py
│   └── unet3d.py
│
├── training/
│   ├── losses.py
│   ├── trainer.py
│   └── metrics.py
│
├── artifacts/                       ← Saved model checkpoints (not committed)
│   ├── model_2d.pt
│   └── model_3d.pt
│
└── results/
    ├── masks/
    │   └── sample_01/               ← One folder per sample
    │       ├── otsu/
    │       ├── yen/
    │       └── bernsen/
    └── metrics/
        ├── global_metrics.csv
        └── pore_stats.csv
```

---

## STEP 1 — RAW XCT INPUT

Input XCT data are provided as ordered TIFF slice stacks:

```
data/raw/sample_01/
    slice_0000.tif
    slice_0001.tif
    slice_0002.tif
    ...
```

Slices **must be named such that alphabetical ordering corresponds to the physical build (Z) direction**.

---

## STEP 2 — XCT PREPROCESSING

Run preprocessing:

```bash
python scripts/run_preprocess.py
```

### Preprocessing Operations

The preprocessing pipeline is designed specifically for **metal XCT data** and applies the following operations in sequence:

1. **3D Median Filtering (3×3×3)**
   - Suppresses speckle noise while preserving sharp defect boundaries

2. **2D Non-Local Means (NLM) Filtering**
   - Noise reduction with edge preservation
   - Noise level estimated automatically from the data

### Output

The output is a normalized 8-bit volume saved slice-wise:

```
data/processed/sample_01/
    slice_0000.tif
    slice_0001.tif
    ...
```

> **Note:** `data/processed/` is the **single shared input source** for all
> subsequent thresholding, comparison, and learning stages. Run this step
> before any other.

---

## STEP 3 — THRESHOLDING METHOD COMPARISON

Three binary segmentation approaches are applied independently to the preprocessed data:

### 1. Global Threshold — Otsu
- Histogram-based global threshold
- Often misses low-contrast pores

### 2. Global Threshold — Yen
- Entropy-based threshold
- Tends to over-segment and introduce edge artifacts

### 3. Local Threshold — Bernsen
- Adaptive local threshold using sliding windows
- Contrast threshold (DCT) computed automatically from matrix noise
- Robust to non-uniform gray-scale intensities

Run comparison:

```bash
python scripts/compare_methods.py
```

### Generated Outputs

Masks are saved **per sample**, grouped by method:

```
results/masks/
    sample_01/
        otsu/
            otsu_slice_0000.tif
            ...
        yen/
            yen_slice_0000.tif
            ...
        bernsen/
            bernsen_slice_0000.tif
            ...

results/metrics/
    global_metrics.csv
    pore_stats.csv
```

---

## STEP 4 — PSEUDO-LABEL GENERATION (WEAK SUPERVISION)

Bernsen's local thresholding results are used to generate **pseudo-labels** for training deep learning models.

Post-processing includes:
- Morphological opening and closing
- Removal of small connected components

Cached pseudo-labels:

```
data/masks/sample_01/
    bernsen_mask_0000.tif
    bernsen_mask_0001.tif
    ...
```

Pseudo-labels are generated automatically during `pipeline.py` if not already cached.

---

## STEP 5 — MODEL TRAINING

Run training:

```bash
python pipeline.py
```

This script:
1. Loads and preprocesses all volumes from `NIST_VOL_DIR` and `PODFAM_VOL_DIR` (set in `config.py`)
2. Generates or loads pseudo-labels
3. Splits volumes into train / val / test sets
4. Trains a **2D slice-wise U-Net** and a **3D volumetric U-Net**
5. Saves best checkpoints

Saved models:

```
artifacts/
    model_2d.pt
    model_3d.pt
```

### config.py parameters

| Parameter | Description |
|---|---|
| `NIST_VOL_DIR` | Path to NIST XCT volumes |
| `PODFAM_VOL_DIR` | Path to PODFAM XCT volumes |
| `VAL_SPLIT` | Fraction of data for validation |
| `TEST_SPLIT` | Fraction of data for test |
| `BATCH_SIZE_2D` | Batch size for 2D training |
| `BATCH_SIZE_3D` | Batch size for 3D training |
| `PATCH_SIZE_3D` | Patch size for 3D model |
| `BERNSEN_RADIUS` | Radius for Bernsen local thresholding |
| `BERNSEN_DCT` | Contrast threshold for Bernsen method |
| `DEVICE` | `cuda` or `cpu` |

---

## STEP 6 — EVALUATION AND VISUALIZATION

Run evaluation:

```bash
python evaluate_2d_vs_3d.py
```

### Quantitative Metrics
- Dice Similarity Coefficient (DSC)
- Intersection over Union (IoU)
- Precision and Recall
- Porosity and pore size distributions

### Qualitative Analysis
- Grayscale + mask overlays
- Slice-by-slice comparison figures
- 3D defect visualization

---

## Known Limitations

- `pipeline.py` loads full volumes into RAM — large datasets may require significant memory
- A minimum of **3 volumes** is required for the train/val/test split in `pipeline.py`
- `src/io.py` reads from `data/processed/` — ensure Step 2 has been run first

---

## Notes on Reproducibility

- Threshold parameters are derived from the data itself
- No fixed segmentation parameters are reused across datasets
- Raw XCT data are excluded from version control via `.gitignore`

---

**Author:** Job George Konnoth Joseph
**Contact:** job-george.konnoth-joseph@student.hv.se
