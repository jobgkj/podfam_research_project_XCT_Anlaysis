"""
metrics.py — Validation and quantitative analysis of XCT defect masks

This module validates binary defect masks and computes slice-wise
quantitative metrics for X-ray Computed Tomography (XCT) data.

Mask convention:
- 1 = solid material (including outside the sample)
- 0 = pore / defect (inside the sample only)

All operations are 2D and slice-wise.
"""

import numpy as np
from skimage.measure import label, regionprops


# ============================================================================
# Validation utilities
# ============================================================================

def validate_mask(mask: np.ndarray) -> None:
    """
    Validate that the input mask is suitable for metric computation.

    Parameters
    ----------
    mask : np.ndarray
        Binary uint8 mask.

    Raises
    ------
    ValueError or TypeError on invalid input.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("Mask must be a NumPy array")

    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape {mask.shape}")

    if mask.dtype != np.uint8:
        raise TypeError("Mask must be of type uint8")

    unique_vals = np.unique(mask)
    if not set(unique_vals).issubset({0, 1}):
        raise ValueError(
            f"Mask must be binary (0/1), got values {unique_vals}"
        )


# ============================================================================
# Global slice-wise metrics
# ============================================================================

def porosity(mask: np.ndarray) -> float:
    """
    Compute porosity for a 2D slice.

    Porosity is defined as:
        pore pixels / total pixels

    Note:
    Outside-sample pixels must already be set to solid (1).
    """
    validate_mask(mask)

    pore_pixels = np.count_nonzero(mask == 0)
    total_pixels = mask.size

    return pore_pixels / total_pixels


def pore_count(mask: np.ndarray, min_area: int = 1) -> int:
    """
    Count the number of pores (connected components).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (1 = solid, 0 = pore)
    min_area : int
        Minimum pore area (in pixels) to be counted
    """
    validate_mask(mask)

    if min_area <= 0:
        raise ValueError("min_area must be positive")

    labeled = label(mask == 0)

    return sum(
        1 for region in regionprops(labeled)
        if region.area >= min_area
    )


# ============================================================================
# Pore-level statistics
# ============================================================================

def pore_properties(mask: np.ndarray, min_area: int = 1) -> dict:
    """
    Extract pore-level properties from a 2D slice.

    Returns
    -------
    dict
        {
            'areas': np.ndarray,
            'equivalent_diameters': np.ndarray
        }
    """
    validate_mask(mask)

    if min_area <= 0:
        raise ValueError("min_area must be positive")

    labeled = label(mask == 0)

    areas = []
    equiv_diams = []

    for region in regionprops(labeled):
        if region.area >= min_area:
            areas.append(region.area)
            equiv_diams.append(region.equivalent_diameter)

    return {
        "areas": np.array(areas, dtype=np.float32),
        "equivalent_diameters": np.array(equiv_diams, dtype=np.float32),
    }


# ============================================================================
# Slice summary (most commonly used entry point)
# ============================================================================

def summarize_slice(mask: np.ndarray, min_area: int = 1) -> dict:
    """
    Compute summary statistics for a single XCT slice.

    Returns
    -------
    dict
        {
            'porosity': float,
            'pore_count': int,
            'mean_pore_area': float,
            'mean_equivalent_diameter': float
        }
    """
    props = pore_properties(mask, min_area=min_area)

    return {
        "porosity": porosity(mask),
        "pore_count": props["areas"].size,
        "mean_pore_area":
            float(np.mean(props["areas"]))
            if props["areas"].size > 0 else 0.0,
        "mean_equivalent_diameter":
            float(np.mean(props["equivalent_diameters"]))
            if props["equivalent_diameters"].size > 0 else 0.0,
    }
