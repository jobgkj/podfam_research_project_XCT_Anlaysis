"""
thresholding.py — Thresholding methods for XCT defect segmentation
with explicit sample masking support.

Conventions:
- Input image: uint8, shape (H, W)
- sample_mask (optional): uint8, shape (H, W), 1 = inside sample, 0 = outside
- Output mask: uint8, shape (H, W)
  - 1 = solid material
  - 0 = pore / defect
"""

import numpy as np
from skimage.filters import threshold_otsu, threshold_yen
from skimage.filters.rank import minimum, maximum
from skimage.morphology import disk


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def _apply_sample_mask(mask: np.ndarray, sample_mask: np.ndarray | None):
    """
    Force outside-sample pixels to solid (1).
    """
    if sample_mask is None:
        return mask

    if sample_mask.shape != mask.shape:
        raise ValueError("sample_mask must have same shape as image/mask")

    if sample_mask.dtype != np.uint8:
        raise TypeError("sample_mask must be uint8")

    # Outside sample → force solid
    mask[sample_mask == 0] = 1
    return mask


# ---------------------------------------------------------------------------
# Global thresholding
# ---------------------------------------------------------------------------
def otsu(img: np.ndarray, sample_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Global Otsu thresholding with optional sample masking.
    """
    if img.dtype != np.uint8:
        raise TypeError("Otsu thresholding expects uint8 input")

    t = threshold_otsu(img)
    mask = (img > t).astype(np.uint8)

    return _apply_sample_mask(mask, sample_mask)


def yen(img: np.ndarray, sample_mask: np.ndarray | None = None) -> np.ndarray:
    """
    Global Yen entropy thresholding with optional sample masking.
    """
    if img.dtype != np.uint8:
        raise TypeError("Yen thresholding expects uint8 input")

    t = threshold_yen(img)
    mask = (img > t).astype(np.uint8)

    return _apply_sample_mask(mask, sample_mask)


# ---------------------------------------------------------------------------
# Local thresholding (Bernsen)
# ---------------------------------------------------------------------------
def bernsen(
    img: np.ndarray,
    radius: int = 5,
    DCT: int = 15,
    sample_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Bernsen local thresholding with optional sample masking.
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

    LCT = local_max - local_min
    Imid = (local_max + local_min) / 2.0

    mask = np.zeros_like(img, dtype=np.uint8)

    low_contrast = LCT < DCT
    mask[low_contrast] = img[low_contrast] > 128

    high_contrast = ~low_contrast
    mask[high_contrast] = img[high_contrast] > Imid[high_contrast]

    return _apply_sample_mask(mask, sample_mask)
