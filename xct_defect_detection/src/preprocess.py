"""
preprocess.py — XCT slice-wise preprocessing

This module implements the complete preprocessing pipeline
for 2D XCT slices of additively manufactured metal parts.

Pipeline (per slice):
1. Median filtering (speckle noise removal)
2. Non-Local Means (edge-preserving denoising)
3. Intensity normalization to uint8 [0, 255]

Design goals:
- 2D only (slice-wise)
- Constant memory usage
- No file I/O side effects
- Suitable for downstream thresholding (Otsu / Yen / Bernsen)
"""

import numpy as np
from scipy.ndimage import median_filter
from skimage.restoration import denoise_nl_means, estimate_sigma


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------
def preprocess_slice(
    img: np.ndarray,
    use_nlm: bool = True,
    median_size: int = 3,
) -> np.ndarray:
    """
    Preprocess a single 2D XCT slice.

    Parameters
    ----------
    img : np.ndarray
        Input 2D XCT slice.
    use_nlm : bool, optional
        Enable non-local means denoising, by default True.
    median_size : int, optional
        Kernel size for median filter, by default 3.

    Returns
    -------
    np.ndarray
        Preprocessed uint8 image in range [0, 255].
    """

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}")

    # Ensure float for processing
    img = img.astype(np.float32)

    # --------------------------------------------------
    # 1. Median filter (remove speckle noise)
    # --------------------------------------------------
    img = median_filter(img, size=median_size)

    # --------------------------------------------------
    # 2. Non-Local Means (edge-preserving denoising)
    # --------------------------------------------------
    if use_nlm:
        sigma = estimate_sigma(img, channel_axis=None)
        img = denoise_nl_means(
            img,
            h=0.6 * sigma,
            patch_size=5,
            patch_distance=21,
            fast_mode=True,
            channel_axis=None,
        )

    # --------------------------------------------------
    # 3. Normalize to uint8
    # --------------------------------------------------
    vmin, vmax = img.min(), img.max()

    if vmax <= vmin:
        raise ValueError("Image has zero intensity range")

    img = (img - vmin) / (vmax - vmin)
    img = (255.0 * img).clip(0, 255).astype(np.uint8)

    return img
