# XCT Defect Segmentation and Evaluation
## for Additively Manufactured Metal Parts

---

## Project Overview

This repository implements a **reproducible, physics-informed workflow**
for defect segmentation and evaluation in **industrial X-ray Computed
Tomography (XCT)** data of additively manufactured metal components.

The pipeline combines **classical image processing–based segmentation**
(global and local thresholding) with **deep learning–based defect
segmentation**, enabling:

- Objective comparison of **global vs local thresholding methods**
- Generation of **weak pseudo-labels** for supervised learning
- Fair benchmarking between **2D and 3D U-Net architectures**
- Quantitative porosity and pore morphology analysis

Raw XCT data are never committed to the repository.

---

## STEP 1 — RAW XCT INPUT

Input XCT data are provided as ordered TIFF slice stacks:
data/raw/sample_01/
slice_0000.tif
slice_0001.tif
slice_0002.tif
...
Slices **must be named such that alphabetical ordering corresponds to the
physical build (Z) direction**.

---

## STEP 2 — XCT PREPROCESSING

Run preprocessing:

```bash
python scripts/run_preprocess.py
Preprocessing operations
The preprocessing pipeline is designed specifically for metal XCT data
and applies the following operations in sequence:


3D median filtering (3×3×3)

Suppresses speckle noise while preserving sharp defect boundaries



2D Non-Local Means (NLM) filtering

Noise reduction with edge preservation
Noise level estimated automatically from the data



The output is a normalized 8‑bit volume saved slice‑wise:
data/processed/sample_01/
    slice_0000.tif
    slice_0001.tif
    ...
This preprocessed data is the single shared input source for all
thresholding, comparison, and learning stages.

STEP 3 — THRESHOLDING METHOD COMPARISON
Three binary segmentation approaches are applied independently to the
same preprocessed data:
1. Global Threshold — Otsu

Histogram-based global threshold
Often misses low-contrast pores

2. Global Threshold — Yen

Entropy-based threshold
Tends to over-segment and introduce edge artifacts

3. Local Threshold — Bernsen

Adaptive local threshold using sliding windows
Contrast threshold (DCT) computed automatically from matrix noise
Robust to non-uniform gray-scale intensities

These methods are evaluated against the original grayscale data to
quantify segmentation bias.
Run comparison:
python scripts/compare_methods.py
Generated outputs:
results/masks/
    otsu/
    yen/
    bernsen/

results/metrics/
    global_metrics.csv
    pore_stats.csv
STEP 4 — PSEUDO-LABEL GENERATION (WEAK SUPERVISION)
Bernsen’s local thresholding results are used to generate pseudo-labels
for training deep learning models.
Post-processing includes:

Morphological opening and closing
Removal of small connected components

Cached pseudo-labels:
data/masks/sample_01/
    bernsen_mask_0000.tif
    bernsen_mask_0001.tif
    ...
data/masks/sample_01/
    bernsen_mask_0000.tif
    bernsen_mask_0001.tif
    ...
python pipeline.py
Saved models:
artifacts/
    model_2d.pt
    model_3d.pt
STEP 7 — EVALUATION AND VISUALIZATION
Quantitative metrics:

Dice Similarity Coefficient (DSC)
Intersection over Union (IoU)
Porosity and pore size distributions

Qualitative analysis:

Grayscale + mask overlays
Slice-by-slice comparison figures
3D defect visualization

Evaluation:
python evaluate_2d_vs_3d.py
Notes on Reproducibility

Threshold parameters are derived from the data itself
No fixed segmentation parameters are reused across datasets
Raw XCT data are excluded from version control


Author: Job George Konnoth Joseph
Contact: job-george.konnoth-joseph@student.hv.se
