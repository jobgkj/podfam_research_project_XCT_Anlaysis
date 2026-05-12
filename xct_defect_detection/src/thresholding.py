"""
=============================================================================
thresholding.py — Thresholding methods for XCT defect segmentation
=============================================================================

Conventions
-----------
- Input image  : uint8, shape (H, W)
- sample_mask  : uint8, shape (H, W), optional
                 1 = inside sample, 0 = outside sample
- Output mask  : uint8, shape (H, W)
                 1 = pore / defect  (dark pixels — foreground)
                 0 = solid material (bright pixels — background)

This convention is consistent with metrics.py and io.py.

Design goals
------------
- Config-driven defaults, all overridable per call
- GUI-friendly: get_default_params() for field pre-population
- No in-place mutation of inputs
=============================================================================
"""

import numpy as np
from skimage.filters import threshold_otsu, threshold_yen
from skimage.filters.rank import minimum, maximum
from skimage.morphology import disk

import config


# =============================================================================
# GUI helper
# =============================================================================

def get_default_params() -> dict:
    """
    Return current thresholding defaults from config.

    GUI can call this to pre-populate its input fields,
    then pass updated values back to bernsen().

    Returns
    -------
    dict with keys:
        bernsen_radius              : int
        bernsen_dct                 : int
        bernsen_low_contrast_thresh : int
    """
    return {
        "bernsen_radius":               config.BERNSEN_RADIUS,
        "bernsen_dct":                  config.BERNSEN_DCT,
        "bernsen_low_contrast_thresh":  config.BERNSEN_LOW_CONTRAST_THRESH,
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _validate_image(img: np.ndarray) -> None:
    """
    Validate that img is a 2D uint8 array.

    Raises
    ------
    TypeError  — if not np.ndarray or not uint8
    ValueError — if not 2D
    """
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")

    if img.dtype != np.uint8:
        raise TypeError(
            f"Thresholding expects uint8 input, got {img.dtype}. "
            "Run preprocess_slice() first."
        )

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")


def _validate_sample_mask(
    sample_mask: np.ndarray,
    img_shape: tuple,
) -> None:
    """
    Validate the optional sample mask.

    Raises
    ------
    TypeError  — if not uint8
    ValueError — if shape does not match image
    """
    if sample_mask.dtype != np.uint8:
        raise TypeError(
            f"sample_mask must be uint8, got {sample_mask.dtype}."
        )

    if sample_mask.shape != img_shape:
        raise ValueError(
            f"sample_mask shape {sample_mask.shape} does not match "
            f"image shape {img_shape}."
        )


def _apply_sample_mask(
    mask:        np.ndarray,
    sample_mask: "np.ndarray | None",
) -> np.ndarray:
    """
    Force outside-sample pixels to solid (0).

    Works on a copy — does not mutate the input mask.
    """
    if sample_mask is None:
        return mask

    mask = mask.copy()
    mask[sample_mask == 0] = 0   # outside sample → solid, never a pore
    return mask


# =============================================================================
# Global thresholding
# =============================================================================

def otsu(
    img:         np.ndarray,
    sample_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Global Otsu thresholding.

    Pixels at or below the threshold are marked as pores (1).
    Pixels above the threshold are marked as solid (0).

    Parameters
    ----------
    img : np.ndarray
        2D uint8 input image.
    sample_mask : np.ndarray, optional
        uint8 mask (1 = inside sample, 0 = outside).
        Outside-sample pixels are forced to solid (0).

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = pore, 0 = solid.
    """
    _validate_image(img)

    if sample_mask is not None:
        _validate_sample_mask(sample_mask, img.shape)

    t    = threshold_otsu(img)
    mask = (img <= t).astype(np.uint8)   # dark pixels = pores

    return _apply_sample_mask(mask, sample_mask)


def yen(
    img:         np.ndarray,
    sample_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Global Yen entropy thresholding.

    Pixels at or below the threshold are marked as pores (1).
    Pixels above the threshold are marked as solid (0).

    Parameters
    ----------
    img : np.ndarray
        2D uint8 input image.
    sample_mask : np.ndarray, optional
        uint8 mask (1 = inside sample, 0 = outside).
        Outside-sample pixels are forced to solid (0).

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = pore, 0 = solid.
    """
    _validate_image(img)

    if sample_mask is not None:
        _validate_sample_mask(sample_mask, img.shape)

    t    = threshold_yen(img)
    mask = (img <= t).astype(np.uint8)   # dark pixels = pores

    return _apply_sample_mask(mask, sample_mask)


# =============================================================================
# Local thresholding — Bernsen
# =============================================================================

def bernsen(
    img:                    np.ndarray,
    radius:                 int                  = None,
    DCT:                    int                  = None,
    low_contrast_thresh:    int                  = None,
    sample_mask:            "np.ndarray | None"  = None,
) -> np.ndarray:
    """
    Bernsen adaptive local thresholding.

    For each pixel:
    - Compute local contrast (LCT) = local_max - local_min
    - If LCT < DCT (low contrast): classify by fixed threshold
    - If LCT >= DCT (high contrast): classify by local midpoint

    Pixels at or below the local threshold = pore (1).
    Pixels above the local threshold       = solid (0).

    Parameters
    ----------
    img : np.ndarray
        2D uint8 input image.
    radius : int, optional
        Radius of the local neighbourhood disk.
        Defaults to config.BERNSEN_RADIUS.
    DCT : int, optional
        Minimum contrast threshold. Regions below this are
        treated as homogeneous (low contrast).
        Defaults to config.BERNSEN_DCT.
    low_contrast_thresh : int, optional
        Fixed intensity threshold for low-contrast regions.
        Defaults to config.BERNSEN_LOW_CONTRAST_THRESH.
    sample_mask : np.ndarray, optional
        uint8 mask (1 = inside sample, 0 = outside).
        Outside-sample pixels are forced to solid (0).

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = pore, 0 = solid.
    """
    # ------------------------------------------------------------------
    # Fall back to config for any unset parameters
    # ------------------------------------------------------------------
    if radius              is None: radius              = config.BERNSEN_RADIUS
    if DCT                 is None: DCT                 = config.BERNSEN_DCT
    if low_contrast_thresh is None: low_contrast_thresh = config.BERNSEN_LOW_CONTRAST_THRESH

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    _validate_image(img)

    if sample_mask is not None:
        _validate_sample_mask(sample_mask, img.shape)

    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}.")

    if DCT < 0:
        raise ValueError(f"DCT must be non-negative, got {DCT}.")

    if not (0 <= low_contrast_thresh <= 255):
        raise ValueError(
            f"low_contrast_thresh must be in [0, 255], got {low_contrast_thresh}."
        )

    # ------------------------------------------------------------------
    # Local min / max via rank filters
    # ------------------------------------------------------------------
    se        = disk(radius)
    local_min = minimum(img, se)
    local_max = maximum(img, se)

    LCT  = (local_max - local_min).astype(np.float32)
    Imid = ((local_max.astype(np.float32) + local_min.astype(np.float32)) / 2.0)

    # ------------------------------------------------------------------
    # Bernsen decision
    # ------------------------------------------------------------------
    mask = np.zeros_like(img, dtype=np.uint8)

    low_contrast  = LCT < DCT
    high_contrast = ~low_contrast

    # Low contrast — use fixed global threshold
    mask[low_contrast] = (img[low_contrast] <= low_contrast_thresh).astype(np.uint8)

    # High contrast — use local midpoint
    mask[high_contrast] = (img[high_contrast] <= Imid[high_contrast]).astype(np.uint8)

    return _apply_sample_mask(mask, sample_mask)
