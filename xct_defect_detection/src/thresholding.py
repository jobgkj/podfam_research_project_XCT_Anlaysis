"""
thresholding.py — Classical thresholding methods for XCT defect segmentation

This module implements global and local thresholding algorithms
to convert a preprocessed 2D XCT slice into a binary defect mask.

Conventions:
- Input image must be uint8, shape (H, W)
- Output mask is uint8, shape (H, W)
- 1 = solid material
- 0 = pore / defect
"""

import numpy as np
from skimage.filters import threshold_otsu, threshold_yen
from skimage.filters.rank import minimum, maximum
from skimage.morphology import disk


# ---------------------------------------------------------------------------
# Global thresholding methods
# ---------------------------------------------------------------------------

def otsu(img: np.ndarray) -> np.ndarray:
    """
    Global Otsu thresholding.

    Parameters
    ----------
    img : np.ndarray
        Preprocessed uint8 image.

    Returns
    -------
    np.ndarray
        Binary uint8 mask (1 = solid, 0 = pore).
    """
    if img.dtype != np.uint8:
        raise TypeError("Otsu thresholding expects uint8 input")

    t = threshold_otsu(img)
    return (img > t).astype(np.uint8)


def yen(img: np.ndarray) -> np.ndarray:
    """
    Global Yen entropy-based thresholding.

    Parameters
    ----------
    img : np.ndarray
        Preprocessed uint8 image.

    Returns
    -------
    np.ndarray
        Binary uint8 mask (1 = solid, 0 = pore).
    """
    if img.dtype != np.uint8:
        raise TypeError("Yen thresholding expects uint8 input")

    t = threshold_yen(img)
    return (img > t).astype(np.uint8)


# ---------------------------------------------------------------------------
# Local thresholding method
# ---------------------------------------------------------------------------

def bernsen(
    img: np.ndarray,
    radius: int = 5,
    DCT: int = 15,
) -> np.ndarray:
    """
    Bernsen local thresholding.

    For each pixel, a local circular window is used to compute
    the local contrast and mid-gray value.

    Parameters
    ----------
    img : np.ndarray
        Preprocessed uint8 image.
    radius : int
        Radius of the local window (in pixels).
    DCT : int
        Local contrast threshold.

    Returns
    -------
    np.ndarray
        Binary uint8 mask (1 = solid, 0 = pore).
    """
    if img.dtype != np.uint8:
        raise TypeError("Bernsen thresholding expects uint8 input")

    if radius <= 0:
        raise ValueError("radius must be positive")

    if DCT < 0:
        raise ValueError("DCT must be non-negative")

    se = disk(radius)

    local_min = minimum(img, se)
    local_max = maximum(img, se)

    # Local contrast
    LCT = local_max - local_min

    # Local midpoint
    Imid = (local_max + local_min) / 2.0

    out = np.zeros_like(img, dtype=np.uint8)

    # Low contrast region: fallback to fixed threshold
    low_contrast = LCT < DCT
    out[low_contrast] = img[low_contrast] > 128

    # High contrast region: local threshold
    high_contrast = ~low_contrast
    out[high_contrast] = img[high_contrast] > Imid[high_contrast]

    return out.astype(np.uint8)
