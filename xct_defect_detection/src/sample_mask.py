"""
=============================================================================
sample_mask.py — Automatic circular sample mask detection for XCT data
=============================================================================

Detects the cylindrical sample boundary in XCT slices and generates
a binary sample mask:
    1 = inside sample  (valid region for thresholding)
    0 = outside sample (air background — forced to solid in all masks)

This prevents air background pixels (dark, outside the cylinder) from
being misclassified as pores, which was causing:
  - Over-estimated porosity
  - Inverted-looking 3D visualisations

Method
------
1. Threshold the image with a low global threshold to separate
   air (very dark) from sample (bright)
2. Find the largest connected component — this is the sample
3. Fit a circle using the component's centroid and equivalent radius
4. Return a filled circular mask

Usage
-----
    from src.sample_mask import detect_sample_mask
    from src.thresholding import bernsen

    sample_mask  = detect_sample_mask(preprocessed_slice)
    mask         = bernsen(preprocessed_slice, sample_mask=sample_mask)

Reference
---------
Kim et al. (2017), Additive Manufacturing.
doi:10.1016/j.addma.2017.06.011
=============================================================================
"""

import warnings
import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, disk

import config


# =============================================================================
# GUI helper
# =============================================================================

def get_default_params() -> dict:
    """
    Return current sample mask defaults from config.

    Returns
    -------
    dict with keys:
        sample_mask_erosion_radius : int
        sample_mask_air_threshold  : int
    """
    return {
        "sample_mask_erosion_radius": config.SAMPLE_MASK_EROSION_RADIUS,
        "sample_mask_air_threshold":  config.SAMPLE_MASK_AIR_THRESHOLD,
    }


# =============================================================================
# Core function
# =============================================================================

def detect_sample_mask(
    img:             np.ndarray,
    erosion_radius:  int = None,
    air_threshold:   int = None,
    return_circle:   bool = True,
) -> np.ndarray:
    """
    Detect the circular sample boundary and return a binary sample mask.

    Parameters
    ----------
    img : np.ndarray
        2D uint8 preprocessed XCT slice.
        Dark background (air) + bright sample (metal).
    erosion_radius : int, optional
        Erode the detected boundary by this many pixels to avoid
        including partial-volume edge pixels in the valid region.
        Defaults to config.SAMPLE_MASK_EROSION_RADIUS.
    air_threshold : int, optional
        Pixels below this value are classified as air (background).
        Defaults to config.SAMPLE_MASK_AIR_THRESHOLD.
    return_circle : bool
        If True  → fit a filled circle to the detected region (recommended).
                   More robust against slice-to-slice variation.
        If False → return the raw largest-component mask (exact boundary).

    Returns
    -------
    np.ndarray
        uint8 mask, same shape as img:
            1 = inside sample
            0 = outside sample (air background)

    Raises
    ------
    ValueError
        If no sample region can be detected.
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")
    if img.dtype != np.uint8:
        raise TypeError(f"Expected uint8 image, got {img.dtype}.")
    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")

    if erosion_radius is None: erosion_radius = config.SAMPLE_MASK_EROSION_RADIUS
    if air_threshold  is None: air_threshold  = config.SAMPLE_MASK_AIR_THRESHOLD

    h, w = img.shape

    # ------------------------------------------------------------------
    # Step 1 — Threshold to separate air from sample
    # Dark air background will be below air_threshold
    # ------------------------------------------------------------------
    binary = img > air_threshold   # True = sample, False = air

    # ------------------------------------------------------------------
    # Step 2 — Fill holes (pores inside sample should be inside mask)
    # ------------------------------------------------------------------
    filled = binary_fill_holes(binary)

    # ------------------------------------------------------------------
    # Step 3 — Find largest connected component = sample cylinder
    # ------------------------------------------------------------------
    labeled  = label(filled)
    regions  = regionprops(labeled)

    if not regions:
        raise ValueError(
            "detect_sample_mask: no regions found. "
            f"Check that air_threshold={air_threshold} is appropriate "
            "for your data (sample should be brighter than air)."
        )

    # Largest region by area = the sample
    sample_region = max(regions, key=lambda r: r.area)

    if return_circle:
        # ------------------------------------------------------------------
        # Step 4a — Fit a circle using centroid + equivalent radius
        # More robust than raw mask — handles incomplete slices at stack edges
        # ------------------------------------------------------------------
        cy, cx = sample_region.centroid
        radius = np.sqrt(sample_region.area / np.pi)

        yi, xi  = np.ogrid[:h, :w]
        circle  = ((xi - cx)**2 + (yi - cy)**2) <= radius**2
        sample_mask = circle.astype(np.uint8)

    else:
        # ------------------------------------------------------------------
        # Step 4b — Use raw component mask
        # ------------------------------------------------------------------
        sample_mask = (labeled == sample_region.label).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 5 — Erode boundary to exclude partial-volume edge pixels
    # Edge pixels mix air and metal intensity → unreliable for thresholding
    # ------------------------------------------------------------------
    if erosion_radius > 0:
        eroded      = binary_erosion(
            sample_mask.astype(bool),
            footprint=disk(erosion_radius)
        )
        sample_mask = eroded.astype(np.uint8)

    # Sanity check
    coverage = sample_mask.sum() / sample_mask.size
    if coverage < 0.05:
        warnings.warn(
            f"detect_sample_mask: sample mask covers only {coverage:.1%} "
            "of the image — mask may be incorrect. "
            f"Try lowering air_threshold (currently {air_threshold})."
        )
    elif coverage > 0.98:
        warnings.warn(
            f"detect_sample_mask: sample mask covers {coverage:.1%} "
            "of the image — background may not be detected. "
            f"Try raising air_threshold (currently {air_threshold})."
        )

    return sample_mask


# =============================================================================
# Stack-level mask (consistent circle across all slices)
# =============================================================================

def detect_sample_mask_stack(
    tiff_files:     list,
    n_sample_slices: int = 5,
    erosion_radius:  int = None,
    air_threshold:   int = None,
) -> tuple:
    """
    Compute a single consistent circular mask for an entire TIFF stack.

    Averages the detected circle parameters (centre + radius) across
    n_sample_slices evenly-spaced slices to produce one stable mask
    that can be applied to every slice in the stack.

    Parameters
    ----------
    tiff_files : list of Path
        Sorted list of TIFF file paths for one sample.
    n_sample_slices : int
        Number of slices to sample for circle detection.
    erosion_radius : int, optional
        Defaults to config.SAMPLE_MASK_EROSION_RADIUS.
    air_threshold : int, optional
        Defaults to config.SAMPLE_MASK_AIR_THRESHOLD.

    Returns
    -------
    tuple (cx, cy, radius, erosion_radius)
        Circle parameters — use build_circle_mask() to generate the mask.
    """
    import tifffile as tiff
    from src.preprocess import preprocess_slice

    if erosion_radius is None: erosion_radius = config.SAMPLE_MASK_EROSION_RADIUS
    if air_threshold  is None: air_threshold  = config.SAMPLE_MASK_AIR_THRESHOLD

    n       = len(tiff_files)
    indices = np.linspace(n // 4, 3 * n // 4, n_sample_slices, dtype=int)

    cxs, cys, radii = [], [], []

    for idx in indices:
        try:
            raw  = tiff.imread(tiff_files[idx]).astype(np.float32)
            if raw.ndim != 2:
                continue
            prep = preprocess_slice(raw)

            binary = prep > air_threshold
            filled = binary_fill_holes(binary)
            labeled = label(filled)
            regions = regionprops(labeled)

            if not regions:
                continue

            region = max(regions, key=lambda r: r.area)
            cy, cx = region.centroid
            radius = np.sqrt(region.area / np.pi)

            cxs.append(cx)
            cys.append(cy)
            radii.append(radius)

        except Exception as e:
            warnings.warn(f"Skipping slice {idx} for mask detection: {e}")
            continue

    if not cxs:
        raise ValueError(
            "detect_sample_mask_stack: could not detect sample circle "
            "in any sampled slice."
        )

    return (
        float(np.mean(cxs)),
        float(np.mean(cys)),
        float(np.mean(radii)),
        erosion_radius,
    )


def build_circle_mask(
    h:              int,
    w:              int,
    cx:             float,
    cy:             float,
    radius:         float,
    erosion_radius: int = 0,
) -> np.ndarray:
    """
    Build a filled circular uint8 mask given circle parameters.

    Parameters
    ----------
    h, w           : image height and width
    cx, cy         : circle centre (x, y)
    radius         : circle radius in pixels
    erosion_radius : shrink mask by this many pixels (edge exclusion)

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = inside sample, 0 = outside.
    """
    yi, xi      = np.ogrid[:h, :w]
    circle      = ((xi - cx)**2 + (yi - cy)**2) <= radius**2
    sample_mask = circle.astype(np.uint8)

    if erosion_radius > 0:
        eroded      = binary_erosion(
            sample_mask.astype(bool),
            footprint=disk(erosion_radius)
        )
        sample_mask = eroded.astype(np.uint8)

    return sample_mask