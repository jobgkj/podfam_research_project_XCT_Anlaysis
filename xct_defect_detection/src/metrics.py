"""
=============================================================================
metrics.py — Validation and quantitative analysis of XCT defect masks
=============================================================================

Mask convention
---------------
    1 = pore / defect  (foreground)
    0 = solid material (background)

This matches the output of otsu(), yen(), and bernsen() in thresholding.py,
and the uint8 masks saved by io.py (values 0 or 1, NOT 0 or 255).

All operations are 2D and slice-wise.
=============================================================================
"""

import numpy as np
from skimage.measure import label, regionprops

import config


# =============================================================================
# Validation
# =============================================================================

def validate_mask(mask: np.ndarray) -> None:
    """
    Validate that the input mask is suitable for metric computation.

    Parameters
    ----------
    mask : np.ndarray
        Binary uint8 mask. Values must be 0 or 1.

    Raises
    ------
    TypeError
        If mask is not a NumPy array or not uint8.
    ValueError
        If mask is not 2D or not binary (0/1).
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a NumPy array.")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}.")

    if mask.dtype != np.uint8:
        raise TypeError(
            f"Mask must be dtype uint8, got {mask.dtype}."
        )

    unique_vals = np.unique(mask)
    if not set(unique_vals).issubset({0, 1}):
        hint = (
            " Mask contains 0/255 — did you forget to divide by 255?"
            if set(unique_vals).issubset({0, 255})
            else ""
        )
        raise ValueError(
            f"Mask must be binary (0/1), got values {unique_vals}.{hint}"
        )


# =============================================================================
# Internal helpers (no validation — called only after validate_mask)
# =============================================================================

def _porosity(mask: np.ndarray) -> float:
    """
    Compute raw porosity without validation.

    Porosity = pore pixels (1) / total pixels.

    Note: counts ALL pore pixels regardless of min_area.
    This is intentional — porosity is a global area fraction,
    not a filtered pore count.
    """
    return float(np.count_nonzero(mask == 1) / mask.size)


def _labeled_regions(mask: np.ndarray, min_area: int):
    """
    Return regionprops for connected pore components >= min_area.
    Shared by pore_count, pore_properties, and summarize_slice
    to avoid recomputing label() multiple times.
    """
    labeled = label(mask == 1)
    return [r for r in regionprops(labeled) if r.area >= min_area]


# =============================================================================
# Public — global slice metrics
# =============================================================================

def porosity(mask: np.ndarray) -> float:
    """
    Compute porosity for a 2D slice.

    Porosity = pore pixels / total pixels.

    Note: all pore pixels are counted regardless of min_area.
    """
    validate_mask(mask)
    return _porosity(mask)


def pore_count(mask: np.ndarray, min_area: int = None) -> int:
    """
    Count connected pore components >= min_area pixels.

    Parameters
    ----------
    mask : np.ndarray
        Binary uint8 mask (1 = pore, 0 = solid).
    min_area : int, optional
        Minimum pore size in pixels.
        Defaults to config.MIN_DEFECT_SIZE.
    """
    if min_area is None:
        min_area = config.MIN_DEFECT_SIZE

    if min_area <= 0:
        raise ValueError("min_area must be positive.")

    validate_mask(mask)
    return len(_labeled_regions(mask, min_area))


# =============================================================================
# Public — pore-level properties
# =============================================================================

def pore_properties(mask: np.ndarray, min_area: int = None) -> dict:
    """
    Extract per-pore properties from a 2D slice.

    Parameters
    ----------
    mask : np.ndarray
        Binary uint8 mask (1 = pore, 0 = solid).
    min_area : int, optional
        Minimum pore size in pixels.
        Defaults to config.MIN_DEFECT_SIZE.

    Returns
    -------
    dict
        {
            'areas'                 : np.ndarray (float32),
            'equivalent_diameters'  : np.ndarray (float32),
            'areas_um2'             : np.ndarray (float32),  # physical units
            'equivalent_diameters_um': np.ndarray (float32), # physical units
        }
    """
    if min_area is None:
        min_area = config.MIN_DEFECT_SIZE

    if min_area <= 0:
        raise ValueError("min_area must be positive.")

    validate_mask(mask)

    regions = _labeled_regions(mask, min_area)

    areas       = np.array([r.area                  for r in regions], dtype=np.float32)
    equiv_diams = np.array([r.equivalent_diameter   for r in regions], dtype=np.float32)

    px = config.PIXEL_SIZE_UM

    return {
        "areas":                    areas,
        "equivalent_diameters":     equiv_diams,
        "areas_um2":                areas       * px ** 2,
        "equivalent_diameters_um":  equiv_diams * px,
    }


# =============================================================================
# Public — slice summary (main entry point for pipeline)
# =============================================================================

def summarize_slice(
    mask:          np.ndarray,
    min_area:      int   = None,
    pixel_size_um: float = None,
) -> dict:
    """
    Compute summary statistics for a single XCT slice.

    Parameters
    ----------
    mask : np.ndarray
        Binary uint8 mask (1 = pore, 0 = solid).
    min_area : int, optional
        Minimum pore size in pixels to include in stats.
        Defaults to config.MIN_DEFECT_SIZE.
    pixel_size_um : float, optional
        Physical pixel size in micrometres.
        Defaults to config.PIXEL_SIZE_UM.

    Returns
    -------
    dict
        {
            'porosity'                  : float  — all pore pixels / total pixels
            'pore_count'                : int    — pores >= min_area
            'mean_pore_area_px'         : float  — mean pore area (pixels)
            'mean_pore_area_um2'        : float  — mean pore area (µm²)
            'mean_equivalent_diameter_px': float — mean equiv. diameter (pixels)
            'mean_equivalent_diameter_um': float — mean equiv. diameter (µm)
        }

    Note
    ----
    porosity counts ALL pore pixels regardless of min_area.
    pore_count and size statistics only include pores >= min_area pixels.
    These are intentionally different — porosity is a global area fraction.
    """
    if min_area is None:
        min_area = config.MIN_DEFECT_SIZE

    if pixel_size_um is None:
        pixel_size_um = config.PIXEL_SIZE_UM

    # Validate once — internal helpers skip re-validation
    validate_mask(mask)

    regions = _labeled_regions(mask, min_area)

    areas       = np.array([r.area                for r in regions], dtype=np.float32)
    equiv_diams = np.array([r.equivalent_diameter for r in regions], dtype=np.float32)

    has_pores = areas.size > 0

    return {
        "porosity":                     _porosity(mask),
        "pore_count":                   areas.size,
        "mean_pore_area_px":            float(np.mean(areas))       if has_pores else 0.0,
        "mean_pore_area_um2":           float(np.mean(areas))       * pixel_size_um ** 2
                                        if has_pores else 0.0,
        "mean_equivalent_diameter_px":  float(np.mean(equiv_diams)) if has_pores else 0.0,
        "mean_equivalent_diameter_um":  float(np.mean(equiv_diams)) * pixel_size_um
                                        if has_pores else 0.0,
    }
