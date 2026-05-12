"""
I/O utilities for streaming 2D XCT slice processing
and binary mask generation.

This module reads individual 2D TIFF slices from:
    data/raw/<sample_name>/

Generates segmentation masks using classical
thresholding methods and writes them to:
    results/masks/<method>/<sample_name>/

Design goals:
- 2D slice-wise only
- Constant memory usage (no full volume loading)
- GitHub-safe (no raw data committed)
"""

from pathlib import Path
import numpy as np
import tifffile as tiff

from src.thresholding import otsu, yen, bernsen


# --------------------------------------------------
def _normalize_to_uint8(img):
    """
    Normalize a 2D image to uint8 [0, 255].
    """
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()

    if vmax <= vmin:
        raise ValueError("Image has zero intensity range")

    img = (img - vmin) / (vmax - vmin)
    return (255.0 * img).clip(0, 255).astype(np.uint8)


# --------------------------------------------------
def load_and_generate_masks(
    repo_root,
    sample_name,
    bernsen_radius=5,
    bernsen_dct=15,
):
    """
    Stream 2D XCT slices from disk, generate segmentation
    masks, and save them to disk.

    Parameters
    ----------
    repo_root : Path or str
        Root directory of the git repository.
    sample_name : str
        Name of the sample folder under data/raw/.
    bernsen_radius : int
        Radius (in pixels) for Bernsen local thresholding.
    bernsen_dct : int
        Contrast threshold (DCT) for Bernsen method.
    """

    repo_root = Path(repo_root)
    raw_dir = repo_root / "data" / "raw" / sample_name

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    # Output directories
    out_dirs = {
        "otsu": repo_root / "results" / "masks" / "otsu" / sample_name,
        "yen": repo_root / "results" / "masks" / "yen" / sample_name,
        "bernsen": repo_root / "results" / "masks" / "bernsen" / sample_name,
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Collect TIFF files (sorted for correct Z-order)
    tiff_files = sorted(
        list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff"))
    )

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {raw_dir}")

    # Stream slices one-by-one
    for idx, f in enumerate(tiff_files):

        try:
            img = tiff.imread(f)
        except Exception as e:
            raise IOError(f"Failed to read TIFF file {f.name}: {e}")

        if img.ndim != 2:
            raise ValueError(f"{f.name} is not a 2D slice (shape={img.shape})")

        if img.dtype != np.uint8:
            img = _normalize_to_uint8(img)

        # Generate masks
        masks = {
            "otsu": otsu(img),
            "yen": yen(img),
            "bernsen": bernsen(
                img,
                radius=bernsen_radius,
                DCT=bernsen_dct,
            ),
        }

        # Save masks
        for method, mask in masks.items():
            tiff.imwrite(
                out_dirs[method] / f"{method}_{idx:04d}.tif",
                mask,
            )
