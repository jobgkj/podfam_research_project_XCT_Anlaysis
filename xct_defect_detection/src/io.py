"""
=============================================================================
I/O utilities for streaming 2D XCT slice processing
and binary mask generation.

Directory layout
----------------
Input  (raw):       data/raw/<sample_name>/<slice>.tif
Input  (processed): data/processed/<sample_name>/<slice>.tif
Output (masks):     data/masks/<sample_name>/<method>/<slice>.tif

All output filenames match the input filename exactly.

Design goals
------------
- 2D slice-wise only
- Constant memory usage (no full volume loading)
- GitHub-safe (no raw data committed)
- GUI-friendly: all parameters readable from config, overridable per call
=============================================================================
"""

from pathlib import Path
import warnings

import numpy as np
import tifffile as tiff

from src.thresholding import otsu, yen, bernsen
import config


# -----------------------------------------------------------------------------
def _normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normalize a 2D image to uint8 [0, 255].

    Returns a zero image if the intensity range is zero
    (blank / empty slice) instead of raising.
    """
    img = img.astype(np.float32)
    vmin, vmax = img.min(), img.max()

    if vmax <= vmin:
        warnings.warn("Slice has zero intensity range — returning blank mask.")
        return np.zeros_like(img, dtype=np.uint8)

    img = (img - vmin) / (vmax - vmin)
    return (255.0 * img).clip(0, 255).astype(np.uint8)


# -----------------------------------------------------------------------------
def get_default_params() -> dict:
    """
    Return current default parameters from config.

    GUI can call this to pre-populate its input fields,
    then pass updated values back to load_and_generate_masks().

    Returns
    -------
    dict with keys:
        repo_root       : Path
        sample_name     : str
        use_processed   : bool
        bernsen_radius  : int
        bernsen_dct     : int
    """
    return {
        "repo_root":      config.REPO_ROOT,
        "sample_name":    config.SAMPLE_NAME,
        "use_processed":  True,
        "bernsen_radius": config.BERNSEN_RADIUS,
        "bernsen_dct":    config.BERNSEN_DCT,
    }


# -----------------------------------------------------------------------------
def load_and_generate_masks(
    repo_root:         "Path | str" = None,
    sample_name:       str          = None,
    use_processed:     bool         = True,
    bernsen_radius:    int          = None,
    bernsen_dct:       int          = None,
    progress_callback: callable     = None,
) -> dict:
    """
    Stream 2D XCT slices from disk, generate segmentation
    masks, and save them to disk.

    All parameters fall back to config.py values if not supplied.
    This makes the function safe to call from a GUI where the user
    may only change a subset of parameters.

    Parameters
    ----------
    repo_root : Path or str, optional
        Root directory of the git repository.
        Defaults to config.REPO_ROOT.
    sample_name : str, optional
        Name of the sample folder under data/raw/ or data/processed/.
        Defaults to config.SAMPLE_NAME.
    use_processed : bool
        If True  → read from data/processed/<sample_name>/  (default)
        If False → read from data/raw/<sample_name>/
    bernsen_radius : int, optional
        Radius (in pixels) for Bernsen local thresholding.
        Defaults to config.BERNSEN_RADIUS.
    bernsen_dct : int, optional
        Contrast threshold (DCT) for Bernsen method.
        Defaults to config.BERNSEN_DCT.
    progress_callback : callable, optional
        Optional function called on each slice with (current, total, filename).
        Use this to update a GUI progress bar.
        Example:
            progress_callback=lambda current, total, fname: print(f"{current}/{total}")

    Returns
    -------
    dict with keys:
        total     : int  — total slices found
        processed : int  — slices successfully processed
        skipped   : int  — slices skipped due to errors
        mask_dir  : Path — root output directory for this sample's masks

    Output
    ------
    Masks are written to:
        data/masks/<sample_name>/otsu/<slice>.tif
        data/masks/<sample_name>/yen/<slice>.tif
        data/masks/<sample_name>/bernsen/<slice>.tif

    Output filenames are identical to the input filenames.
    """

    # ------------------------------------------------------------------
    # Fall back to config values for any unset parameters
    # ------------------------------------------------------------------
    repo_root      = Path(repo_root) if repo_root      is not None else config.REPO_ROOT
    sample_name    = sample_name     if sample_name     is not None else config.SAMPLE_NAME
    bernsen_radius = bernsen_radius  if bernsen_radius  is not None else config.BERNSEN_RADIUS
    bernsen_dct    = bernsen_dct     if bernsen_dct     is not None else config.BERNSEN_DCT

    # ------------------------------------------------------------------
    # Input directory
    # ------------------------------------------------------------------
    if use_processed:
        input_dir = repo_root / "data" / "processed" / sample_name
    else:
        input_dir = repo_root / "data" / "raw" / sample_name

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # ------------------------------------------------------------------
    # Output directories — data/masks/<sample_name>/<method>/
    # ------------------------------------------------------------------
    mask_root = repo_root / "data" / "masks" / sample_name

    out_dirs = {
        "otsu":    mask_root / "otsu",
        "yen":     mask_root / "yen",
        "bernsen": mask_root / "bernsen",
    }

    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Collect TIFF files (sorted for correct Z-order)
    # ------------------------------------------------------------------
    tiff_files = sorted(
        list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    )

    if not tiff_files:
        raise ValueError(f"No TIFF files found in {input_dir}")

    total   = len(tiff_files)
    skipped = 0

    print(f"Found {total} slices in {input_dir}")

    # ------------------------------------------------------------------
    # Stream slices one-by-one
    # ------------------------------------------------------------------
    for idx, f in enumerate(tiff_files):

        # Console progress
        print(f"  Processing {idx + 1}/{total}: {f.name}", end="\r")

        # GUI progress bar hook
        if progress_callback is not None:
            progress_callback(idx + 1, total, f.name)

        # Load
        try:
            img = tiff.imread(f)
        except Exception as e:
            warnings.warn(f"Skipping {f.name} — could not read: {e}")
            skipped += 1
            continue

        # Validate shape
        if img.ndim != 2:
            warnings.warn(
                f"Skipping {f.name} — not a 2D slice (shape={img.shape})"
            )
            skipped += 1
            continue

        # Normalize to uint8 if needed
        if img.dtype != np.uint8:
            img = _normalize_to_uint8(img)

        # Generate masks
        masks = {
            "otsu":    otsu(img),
            "yen":     yen(img),
            "bernsen": bernsen(img, radius=bernsen_radius, DCT=bernsen_dct),
        }

        # Save — preserve original filename exactly
        for method, mask in masks.items():
            tiff.imwrite(
                out_dirs[method] / f.name,
                mask.astype(np.uint8),
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    processed = total - skipped

    print(f"\nDone — {processed}/{total} slices processed.")
    if skipped:
        print(f"  Skipped: {skipped} slice(s) — check warnings above.")
    print(f"  Masks saved to: {mask_root}")

    return {
        "total":     total,
        "processed": processed,
        "skipped":   skipped,
        "mask_dir":  mask_root,
    }
