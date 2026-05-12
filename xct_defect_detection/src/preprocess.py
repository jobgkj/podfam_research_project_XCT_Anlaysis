"""
=============================================================================
preprocess.py — XCT slice-wise preprocessing
=============================================================================

Pipeline (per slice)
--------------------
1. Median filtering       — speckle noise removal
2. Non-Local Means (NLM)  — edge-preserving denoising
3. Intensity normalization — uint8 [0, 255]

Design goals
------------
- 2D only (slice-wise)
- Constant memory usage
- No file I/O side effects
- Config-driven defaults, all overridable per call
- GUI-friendly: get_default_params() for field pre-population
=============================================================================
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from skimage.restoration import denoise_nl_means, estimate_sigma

import config


# =============================================================================
# GUI helper
# =============================================================================

def get_default_params() -> dict:
    """
    Return current preprocessing defaults from config.

    GUI can call this to pre-populate its input fields,
    then pass updated values back to preprocess_slice().

    Returns
    -------
    dict with keys:
        use_nlm        : bool
        median_size    : int
        nlm_h_factor   : float  — h = nlm_h_factor * estimated_sigma
        patch_size     : int
        patch_distance : int
    """
    return {
        "use_nlm":        True,
        "median_size":    config.MEDIAN_KERNEL_SIZE,
        "nlm_h_factor":   config.NLM_H_FACTOR,
        "patch_size":     config.NLM_PATCH_SIZE,
        "patch_distance": config.NLM_PATCH_DIST,
    }


# =============================================================================
# Core preprocessing function
# =============================================================================

def preprocess_slice(
    img:            np.ndarray,
    use_nlm:        bool  = None,
    median_size:    int   = None,
    nlm_h_factor:   float = None,
    patch_size:     int   = None,
    patch_distance: int   = None,
) -> np.ndarray:
    """
    Preprocess a single 2D XCT slice.

    All parameters fall back to config.py values if not supplied.
    This makes the function safe to call from a GUI where the user
    may only change a subset of parameters.

    Parameters
    ----------
    img : np.ndarray
        Input 2D XCT slice. Any numeric dtype accepted.
    use_nlm : bool, optional
        Enable Non-Local Means denoising.
        Defaults to True.
    median_size : int, optional
        Kernel size for median filter.
        Defaults to config.MEDIAN_KERNEL_SIZE.
    nlm_h_factor : float, optional
        NLM filter strength as a multiple of the estimated noise sigma.
        h = nlm_h_factor * sigma  (adaptive — recommended for XCT).
        Defaults to config.NLM_H_FACTOR.
    patch_size : int, optional
        NLM patch size (pixels).
        Defaults to config.NLM_PATCH_SIZE.
    patch_distance : int, optional
        NLM patch search radius (pixels).
        Defaults to config.NLM_PATCH_DIST.

    Returns
    -------
    np.ndarray
        Preprocessed uint8 image in range [0, 255].
        Returns a zero-filled image with a warning if the
        intensity range is zero (blank slice).

    Raises
    ------
    ValueError
        If img is not 2D.
    TypeError
        If img is not a numeric array.
    """

    # ------------------------------------------------------------------
    # Fall back to config for any unset parameters
    # ------------------------------------------------------------------
    if median_size    is None: median_size    = config.MEDIAN_KERNEL_SIZE
    if nlm_h_factor   is None: nlm_h_factor   = config.NLM_H_FACTOR
    if patch_size     is None: patch_size     = config.NLM_PATCH_SIZE
    if patch_distance is None: patch_distance = config.NLM_PATCH_DIST
    if use_nlm        is None: use_nlm        = True

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray, got {type(img).__name__}.")

    if not np.issubdtype(img.dtype, np.number):
        raise TypeError(f"Expected numeric image, got dtype {img.dtype}.")

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")

    # ------------------------------------------------------------------
    # Convert to float32 for all processing
    # ------------------------------------------------------------------
    img = img.astype(np.float32)

    # ------------------------------------------------------------------
    # Step 1 — Median filter (speckle noise removal)
    # ------------------------------------------------------------------
    img = median_filter(img, size=median_size)

    # ------------------------------------------------------------------
    # Step 2 — Non-Local Means (edge-preserving denoising)
    # ------------------------------------------------------------------
    if use_nlm:
        sigma = estimate_sigma(img, channel_axis=None)

        if sigma > 0:
            img = denoise_nl_means(
                img,
                h=nlm_h_factor * sigma,
                patch_size=patch_size,
                patch_distance=patch_distance,
                fast_mode=True,
                channel_axis=None,
            )
        else:
            warnings.warn(
                "Estimated noise sigma is zero — skipping NLM denoising."
            )

    # ------------------------------------------------------------------
    # Step 3 — Normalize to uint8 [0, 255]
    # ------------------------------------------------------------------
    vmin, vmax = img.min(), img.max()

    if vmax <= vmin:
        warnings.warn(
            "Slice has zero intensity range — returning blank slice."
        )
        return np.zeros(img.shape, dtype=np.uint8)

    img = (img - vmin) / (vmax - vmin)
    return (255.0 * img).clip(0, 255).astype(np.uint8)
