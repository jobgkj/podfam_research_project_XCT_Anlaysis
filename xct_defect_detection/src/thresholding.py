"""
=============================================================================
thresholding.py — Thresholding methods for XCT defect segmentation
=============================================================================

Conventions
-----------
- Input image  : uint8, shape (H, W) — must be preprocessed (NLM denoised)
- sample_mask  : uint8, shape (H, W), optional
                 1 = inside sample, 0 = outside sample
- Output mask  : uint8, shape (H, W)
                 1 = pore / defect  (dark pixels — foreground)
                 0 = solid material (bright pixels — background)

This convention is consistent with metrics.py and io.py.

DCT Auto-computation (Kim et al. 2017)
---------------------------------------
The Bernsen DCT parameter is computed per image as:
    DCT = 18 × mean(local std in solid-phase windows)

This is the exact method from:
    Kim et al., Additive Manufacturing, 2017.
    doi:10.1016/j.addma.2017.06.011

The paper measured avg std = 0.847 for Sample 2 after NLM filtering,
giving DCT = 18 × 0.847 ≈ 15.
For noisier images the std will be higher → higher DCT → less over-segmentation.

Set config.BERNSEN_DCT_AUTO = False to use a fixed DCT value instead.
=============================================================================
"""

import warnings
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

    GUI can call this to pre-populate its input fields.

    Returns
    -------
    dict with keys:
        bernsen_radius              : int
        bernsen_dct_auto            : bool
        bernsen_dct_std_multiplier  : int
        bernsen_dct                 : int   (fallback fixed value)
        bernsen_low_contrast_thresh : int
    """
    return {
        "bernsen_radius":               config.BERNSEN_RADIUS,
        "bernsen_dct_auto":             config.BERNSEN_DCT_AUTO,
        "bernsen_dct_std_multiplier":   config.BERNSEN_DCT_STD_MULTIPLIER,
        "bernsen_dct":                  config.BERNSEN_DCT,
        "bernsen_low_contrast_thresh":  config.BERNSEN_LOW_CONTRAST_THRESH,
    }


# =============================================================================
# Internal validation helpers
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
            "Run preprocess_slice() first to produce a uint8 image."
        )

    if img.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {img.shape}.")


def _validate_sample_mask(
    sample_mask: np.ndarray,
    img_shape:   tuple,
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
    mask[sample_mask == 0] = 0
    return mask


# =============================================================================
# Auto-DCT computation (Kim et al. 2017)
# =============================================================================

def compute_dct_from_image(img: np.ndarray) -> int:
    """
    Compute the Bernsen DCT parameter automatically from the image's
    solid-phase noise standard deviation.

    Method from Kim et al. (2017), Additive Manufacturing:
        DCT = 18 × mean(local std of solid-phase windows)

    Solid-phase windows are identified as local regions with mean
    intensity > 128 (bright = solid metal in uint8 XCT images).
    Windows are sampled on a grid across the centre half of the image
    to avoid edge artefacts.

    Parameters
    ----------
    img : np.ndarray
        2D uint8 preprocessed image (after NLM denoising).

    Returns
    -------
    int
        Computed DCT value (minimum config.BERNSEN_DCT_MIN).
        Falls back to config.BERNSEN_DCT if no solid windows found.
    """
    _validate_image(img)

    radius  = config.BERNSEN_DCT_WINDOW_RADIUS
    n_locs  = config.BERNSEN_DCT_N_LOCATIONS
    h, w    = img.shape

    # Sample grid across centre half of image (avoids reconstruction edges)
    ys = np.linspace(h // 4, 3 * h // 4, n_locs, dtype=int)
    xs = np.linspace(w // 4, 3 * w // 4, n_locs, dtype=int)

    stds = []
    for y in ys:
        for x in xs:
            y0 = max(0, y - radius)
            y1 = min(h, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(w, x + radius + 1)

            window = img[y0:y1, x0:x1].astype(np.float32)

            # Only include windows in the solid phase
            # (mean > 128 = bright = solid metal in XCT uint8 images)
            if window.mean() > 128:
                stds.append(float(window.std()))

    if not stds:
        warnings.warn(
            "compute_dct_from_image: no solid-phase windows found "
            f"(all windows have mean ≤ 128). "
            f"Falling back to config.BERNSEN_DCT = {config.BERNSEN_DCT}. "
            "Check that the image is correctly normalised to uint8."
        )
        return config.BERNSEN_DCT

    mean_std = float(np.mean(stds))
    dct      = int(config.BERNSEN_DCT_STD_MULTIPLIER * mean_std)
    dct      = max(dct, config.BERNSEN_DCT_MIN)

    return dct


# =============================================================================
# Global thresholding — Otsu
# =============================================================================

def otsu(
    img:         np.ndarray,
    sample_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Global Otsu thresholding on preprocessed (NLM denoised) uint8 image.

    Pixels at or below the threshold = pore (1).
    Pixels above the threshold       = solid (0).

    Note: Otsu is provided as a comparison baseline only.
    Kim et al. (2017) showed Otsu fails on low-porosity XCT data due to
    non-uniform intensity in the solid phase. Use Bernsen for primary results.

    Parameters
    ----------
    img : np.ndarray
        2D uint8 preprocessed image.
    sample_mask : np.ndarray, optional
        uint8 mask (1 = inside sample, 0 = outside).

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = pore, 0 = solid.
    """
    _validate_image(img)

    if sample_mask is not None:
        _validate_sample_mask(sample_mask, img.shape)

    t    = threshold_otsu(img)
    mask = (img <= t).astype(np.uint8)

    return _apply_sample_mask(mask, sample_mask)


# =============================================================================
# Global thresholding — Yen
# =============================================================================

def yen(
    img:         np.ndarray,
    sample_mask: "np.ndarray | None" = None,
) -> np.ndarray:
    """
    Global Yen entropy thresholding on preprocessed uint8 image.

    Pixels at or below the threshold = pore (1).
    Pixels above the threshold       = solid (0).

    Note: Yen is provided as a comparison baseline only.
    Kim et al. (2017) showed Yen over-segments XCT data due to
    non-uniform intensity at image edges. Use Bernsen for primary results.

    Parameters
    ----------
    img : np.ndarray
        2D uint8 preprocessed image.
    sample_mask : np.ndarray, optional
        uint8 mask (1 = inside sample, 0 = outside).

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = pore, 0 = solid.
    """
    _validate_image(img)

    if sample_mask is not None:
        _validate_sample_mask(sample_mask, img.shape)

    t    = threshold_yen(img)
    mask = (img <= t).astype(np.uint8)

    return _apply_sample_mask(mask, sample_mask)


# =============================================================================
# Local thresholding — Bernsen (primary method)
# =============================================================================

def bernsen(
    img:                 np.ndarray,
    radius:              int                  = None,
    DCT:                 int                  = None,
    low_contrast_thresh: int                  = None,
    sample_mask:         "np.ndarray | None"  = None,
) -> np.ndarray:
    """
    Bernsen adaptive local thresholding — primary segmentation method.

    Implements the exact algorithm from Kim et al. (2017):
        For each pixel:
        - Compute LCT = local_max - local_min within radius
        - If LCT < DCT (low contrast): classify by fixed threshold (128)
        - If LCT ≥ DCT (high contrast): classify by local midpoint (Imid)

    DCT is auto-computed from the image's own noise std if
    config.BERNSEN_DCT_AUTO = True (recommended).

    Pixels at or below the local threshold = pore (1).
    Pixels above the local threshold       = solid (0).

    Parameters
    ----------
    img : np.ndarray
        2D uint8 preprocessed image (after NLM denoising — required for
        accurate DCT estimation and correct segmentation).
    radius : int, optional
        Radius of the local neighbourhood disk in pixels.
        Defaults to config.BERNSEN_RADIUS (= 5, as per Kim et al. 2017).
    DCT : int, optional
        Minimum contrast threshold.
        If None and config.BERNSEN_DCT_AUTO = True:
            auto-computed as 18 × mean(local std of solid phase).
        If None and config.BERNSEN_DCT_AUTO = False:
            uses config.BERNSEN_DCT (fixed fallback).
        Pass an explicit int to override both.
    low_contrast_thresh : int, optional
        Fixed intensity threshold for low-contrast regions.
        Defaults to config.BERNSEN_LOW_CONTRAST_THRESH (= 128).
    sample_mask : np.ndarray, optional
        uint8 mask (1 = inside sample, 0 = outside).
        Outside-sample pixels are forced to solid (0).

    Returns
    -------
    np.ndarray
        uint8 mask — 1 = pore, 0 = solid.

    References
    ----------
    Kim et al. (2017), Additive Manufacturing.
    doi:10.1016/j.addma.2017.06.011
    """

    # ------------------------------------------------------------------
    # Fall back to config for any unset parameters
    # ------------------------------------------------------------------
    if radius              is None: radius              = config.BERNSEN_RADIUS
    if low_contrast_thresh is None: low_contrast_thresh = config.BERNSEN_LOW_CONTRAST_THRESH

    # Auto-compute DCT from image if enabled and not manually overridden
    if DCT is None:
        if config.BERNSEN_DCT_AUTO:
            DCT = compute_dct_from_image(img)
        else:
            DCT = config.BERNSEN_DCT

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
            f"low_contrast_thresh must be in [0, 255], "
            f"got {low_contrast_thresh}."
        )

    # ------------------------------------------------------------------
    # Local min / max via rank filters
    # ------------------------------------------------------------------
    se        = disk(radius)
    local_min = minimum(img, se)
    local_max = maximum(img, se)

    LCT  = (local_max.astype(np.float32) - local_min.astype(np.float32))
    Imid = ((local_max.astype(np.float32) + local_min.astype(np.float32)) / 2.0)

    # ------------------------------------------------------------------
    # Bernsen decision (Kim et al. 2017)
    # ------------------------------------------------------------------
    mask = np.zeros_like(img, dtype=np.uint8)

    low_contrast  = LCT < DCT
    high_contrast = ~low_contrast

    # Low contrast region — use fixed global threshold
    # Solid phase (bright) has intensity > 128 → classified as solid (0)
    mask[low_contrast] = (
        img[low_contrast] <= low_contrast_thresh
    ).astype(np.uint8)

    # High contrast region — use local midpoint
    # Pixels below local midpoint are pores (dark)
    mask[high_contrast] = (
        img[high_contrast] <= Imid[high_contrast]
    ).astype(np.uint8)

    return _apply_sample_mask(mask, sample_mask)
