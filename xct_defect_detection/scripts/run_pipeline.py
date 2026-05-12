"""
run_pipeline.py — End-to-end XCT processing pipeline (2D, slice-wise)

This script:
1. Reads raw 2D XCT slices
2. Preprocesses each slice
3. Segments defects using thresholding
4. Computes slice-wise metrics
5. Saves masks to disk

Run from repository root:
    python scripts/run_pipeline.py
"""

from pathlib import Path
import tifffile as tiff

import config
from src.preprocess import preprocess_slice
from src.thresholding import bernsen, otsu, yen
from src.metrics import summarize_slice


def main():
    # --------------------------------------------------
    # Paths
    # --------------------------------------------------
    raw_dir = config.RAW_DATA_DIR / config.SAMPLE_NAME

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_dir}")

    # Output directories already created by config.py
    out_dirs = {
        "otsu": config.OTSU_MASK_DIR / config.SAMPLE_NAME,
        "yen": config.YEN_MASK_DIR / config.SAMPLE_NAME,
        "bernsen": config.BERNSEN_MASK_DIR / config.SAMPLE_NAME,
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Processing loop (slice-wise, low memory)
    # --------------------------------------------------
    tiff_files = sorted(list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff")))

    all_metrics = []

    for idx, f in enumerate(tiff_files):
        print(f"Processing slice {idx + 1}/{len(tiff_files)}", end="\r")

        # ---- Load raw slice ----
        img_raw = tiff.imread(f)

        # ---- Preprocessing ----
        img_pre = preprocess_slice(img_raw)

        # ---- Thresholding ----
        masks = {
            "otsu": otsu(img_pre),
            "yen": yen(img_pre),
            "bernsen": bernsen(
                img_pre,
                radius=config.BERNSEN_RADIUS,
                DCT=config.BERNSEN_DCT,
            ),
        }

        # ---- Save masks + compute metrics ----
        for method, mask in masks.items():
            tiff.imwrite(
                out_dirs[method] / f"{method}_{idx:04d}.tif",
                mask,
            )

            stats = summarize_slice(mask)
            stats["method"] = method
            stats["slice"] = idx
            all_metrics.append(stats)

    print("\nProcessing complete.")

    # --------------------------------------------------
    # (Optional) aggregate metrics later
    # --------------------------------------------------
    print(f"Computed metrics for {len(all_metrics)} slice–method pairs.")


if __name__ == "__main__":
    main()
