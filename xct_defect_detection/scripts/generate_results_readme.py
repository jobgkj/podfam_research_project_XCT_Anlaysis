"""
=============================================================================
thesis_analysis.py — Complete Thesis Analysis Pipeline
=============================================================================

Runs all 5 tasks for every sample in data/raw/:
  Task 1 — Preprocessing analysis
  Task 2 — Segmentation comparison (Otsu / Yen / Bernsen)
  Task 3 — Porosity & pore analysis
  Task 4 — Patch extraction & augmentation
  Task 5 — 3D visualisation (static PNG)

Each task processes the middle slice of every sample.

Outputs saved to: thesis_outputs/task<N>/

Run from project root:
    python thesis_analysis.py
=============================================================================
"""
from pathlib import Path
import time
import csv
import warnings
import numpy as np
import tifffile as tiff
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter, median_filter
from skimage.filters import threshold_otsu
from skimage.measure import marching_cubes
from skimage.restoration import denoise_nl_means, estimate_sigma

import config
from config import create_dirs
from src.preprocess import preprocess_slice
from src.thresholding import otsu, yen, bernsen
from src.metrics import summarize_slice, pore_properties

# =============================================================================
# Global settings
# =============================================================================

THESIS_ROOT  = config.REPO_ROOT / "thesis_outputs"
PLOT_DPI     = 150
METHODS      = ["otsu", "yen", "bernsen"]
COLORS       = {"otsu": "#e05c5c", "yen": "#5ca8e0", "bernsen": "#5ce08a"}
PORE_CMAP    = ListedColormap(["none", "#ff4444"])

# Task 5 settings
N_SLICES_3D  = 10
MAX_POINTS   = 20_000
RANDOM_SEED  = 42


# =============================================================================
# Directory helpers
# =============================================================================

def out_dir(task: int) -> "Path":
    d = THESIS_ROOT / f"task{task}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_fig(fig, task: int, name: str):
    path = out_dir(task) / name
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"      [Saved] task{task}/{name}")


def dark_ax(ax):
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#333333")


# =============================================================================
# Shared — load middle slice
# =============================================================================

def load_middle_slice(sample_name: str) -> tuple:
    """
    Returns (raw float32 array, mid filename stem).
    """
    raw_dir    = config.RAW_DATA_DIR / sample_name
    tiff_files = sorted(
        list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff"))
    )
    if not tiff_files:
        raise ValueError(f"No TIFF files in {raw_dir}")

    mid_file = tiff_files[len(tiff_files) // 2]
    raw      = tiff.imread(mid_file).astype(np.float32)

    if raw.ndim != 2:
        raise ValueError(f"{mid_file.name} is not a 2D slice")

    return raw, mid_file.stem


def load_subvolume(sample_name: str) -> tuple:
    """
    Load N_SLICES_3D slices centred on the middle for Task 5.
    Returns (prep_volume, mask_volume) as float32 arrays (N, H, W).
    """
    raw_dir    = config.RAW_DATA_DIR / sample_name
    tiff_files = sorted(
        list(raw_dir.glob("*.tif")) + list(raw_dir.glob("*.tiff"))
    )
    if not tiff_files:
        raise ValueError(f"No TIFF files in {raw_dir}")

    total = len(tiff_files)
    mid   = total // 2
    half  = N_SLICES_3D // 2
    start = max(0, mid - half)
    end   = min(total, start + N_SLICES_3D)
    files = tiff_files[start:end]

    prep_slices = []
    mask_slices = []

    for f in files:
        raw = tiff.imread(f).astype(np.float32)
        if raw.ndim != 2:
            continue
        prep = preprocess_slice(raw)
        mask = bernsen(prep)
        prep_slices.append(prep.astype(np.float32) / 255.0)
        mask_slices.append(mask.astype(np.float32))

    if not prep_slices:
        raise ValueError(f"No valid 2D slices for {sample_name}")

    return np.stack(prep_slices, axis=0), np.stack(mask_slices, axis=0)


# =============================================================================
# Preprocessing helpers
# =============================================================================

def step_normalize(slc: np.ndarray) -> np.ndarray:
    lo = np.percentile(slc, config.NORM_LOW_PERCENTILE)
    hi = np.percentile(slc, config.NORM_HIGH_PERCENTILE)
    return np.clip(
        (slc - lo) / (hi - lo + config.NORM_EPS), 0, 1
    ).astype(np.float32)


def step_median(slc: np.ndarray) -> np.ndarray:
    return median_filter(slc, size=config.MEDIAN_KERNEL_SIZE).astype(np.float32)


def step_nlm(slc: np.ndarray) -> np.ndarray:
    sigma = estimate_sigma(slc)
    if sigma <= 0:
        return slc
    return denoise_nl_means(
        slc,
        h=config.NLM_H_FACTOR * sigma,
        patch_size=config.NLM_PATCH_SIZE,
        patch_distance=config.NLM_PATCH_DIST,
        fast_mode=True,
        channel_axis=None,
    ).astype(np.float32)


def compute_snr(slc: np.ndarray) -> float:
    std = slc.std()
    return float(slc.mean() / std) if std > 0 else 0.0


def apply_augmentation(img: np.ndarray, mask: np.ndarray):
    if np.random.random() < config.AUG_FLIP_PROB:
        img, mask = np.fliplr(img), np.fliplr(mask)
    if np.random.random() < config.AUG_ROTATE_PROB:
        k = np.random.choice([1, 2, 3])
        img, mask = np.rot90(img, k), np.rot90(mask, k)
    return img.copy(), mask.copy()


# =============================================================================
# TASK 1 — Preprocessing
# =============================================================================

def task1(sample_name: str, raw: np.ndarray, mid_slice: str) -> dict:
    s1 = step_normalize(raw)
    s2 = step_median(s1)
    s3 = step_nlm(s2)

    steps = {
        "Raw":              raw,
        "Normalised":       s1,
        "Median\nFiltered": s2,
        "NLM\nDenoised":    s3,
    }
    snrs = {k: compute_snr(v) for k, v in steps.items()}

    # ── Pipeline panel ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0d0d0d")
    fig.suptitle(
        f"Preprocessing Pipeline — {sample_name} — Slice: {mid_slice}",
        color="white", fontsize=13, y=1.02
    )
    for ax, (label, slc) in zip(axes, steps.items()):
        ax.imshow(slc, cmap="gray")
        ax.set_title(label, color="white", fontsize=10, pad=6)
        ax.axis("off")
        ax.text(0.5, -0.07,
                f"[{slc.min():.3f}, {slc.max():.3f}]  SNR={compute_snr(slc):.2f}",
                transform=ax.transAxes, ha="center",
                color="#888888", fontsize=8)
    plt.tight_layout()
    save_fig(fig, 1, f"{sample_name}_pipeline_panel.png")

    # ── ROI zoom ─────────────────────────────────────────────────────────────
    h, w     = raw.shape
    y0, x0   = h // 4, w // 4
    rh, rw   = h // 4, w // 4

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), facecolor="#0d0d0d")
    fig.suptitle(f"ROI Zoom — {sample_name}", color="white", fontsize=13)
    for ax, (label, slc) in zip(axes, steps.items()):
        ax.imshow(slc[y0:y0+rh, x0:x0+rw], cmap="gray")
        ax.set_title(label, color="white", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, 1, f"{sample_name}_roi_zoom.png")

    # ── Histograms ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0d0d")
    fig.suptitle(f"Histograms — {sample_name} — Raw vs NLM",
                 color="white", fontsize=13)
    for ax, slc, label, color in zip(
        axes, [raw, s3],
        ["Raw", "NLM Denoised"],
        ["#e05c5c", "#5ca8e0"]
    ):
        dark_ax(ax)
        ax.hist(slc.ravel(), bins=256, color=color, alpha=0.85, edgecolor="none")
        ax.axvline(slc.mean(),      color="white",  lw=1.2, ls="--",
                   label=f"μ={slc.mean():.3f}")
        ax.axvline(np.median(slc),  color="yellow", lw=1.2, ls=":",
                   label=f"med={np.median(slc):.3f}")
        ax.set_title(label, color="white", fontsize=11)
        ax.set_xlabel("Intensity", color="white")
        ax.set_ylabel("Frequency", color="white")
        ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
    plt.tight_layout()
    save_fig(fig, 1, f"{sample_name}_histograms.png")

    # ── NLM tuning ───────────────────────────────────────────────────────────
    h_values = [0.3, 0.6, 1.0, 1.5]
    sigma    = estimate_sigma(s1)
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#0d0d0d")
    fig.suptitle(f"NLM h_factor Tuning — {sample_name}",
                 color="white", fontsize=13)
    for ax, h in zip(axes, h_values):
        d   = denoise_nl_means(s1, h=h*sigma if sigma > 0 else h,
                               patch_size=config.NLM_PATCH_SIZE,
                               patch_distance=config.NLM_PATCH_DIST,
                               fast_mode=True).astype(np.float32)
        snr = compute_snr(d)
        ax.imshow(d, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"h={h}×σ  SNR={snr:.2f}", color="white", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, 1, f"{sample_name}_nlm_tuning.png")

    return {"sample_name": sample_name, "snrs": snrs}


# =============================================================================
# TASK 2 — Segmentation
# =============================================================================

def task2(sample_name: str, preprocessed: np.ndarray,
          mid_slice: str) -> dict:
    masks = {
        "otsu":    otsu(preprocessed),
        "yen":     yen(preprocessed),
        "bernsen": bernsen(preprocessed),
    }
    pore_fracs = {
        m: float(np.sum(mask == 1)) / mask.size
        for m, mask in masks.items()
    }

    # ── Method comparison ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0d0d0d")
    fig.suptitle(
        f"Segmentation Comparison — {sample_name} — Slice: {mid_slice}",
        color="white", fontsize=13, y=1.01
    )
    for col, method in enumerate(METHODS):
        mask = masks[method]
        axes[0, col].imshow(preprocessed, cmap="gray")
        axes[0, col].imshow(mask, cmap=PORE_CMAP, alpha=0.5)
        axes[0, col].set_title(
            f"{method.capitalize()} — Overlay\nPore frac: {pore_fracs[method]:.4f}",
            color="white", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title(f"{method.capitalize()} — Binary",
                               color="white", fontsize=10)
        axes[1, col].axis("off")
    plt.tight_layout()
    save_fig(fig, 2, f"{sample_name}_method_comparison.png")

    # ── Difference maps ──────────────────────────────────────────────────────
    pairs = [("otsu", "yen"), ("otsu", "bernsen"), ("yen", "bernsen")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d0d0d")
    fig.suptitle(f"Method Disagreement — {sample_name}",
                 color="white", fontsize=13)
    for ax, (m1, m2) in zip(axes, pairs):
        diff = (masks[m1] != masks[m2]).astype(np.uint8)
        pct  = diff.sum() / diff.size * 100
        ax.imshow(diff, cmap="hot", vmin=0, vmax=1)
        ax.set_title(f"{m1.capitalize()} vs {m2.capitalize()}\n"
                     f"Disagreement: {pct:.2f}%",
                     color="white", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, 2, f"{sample_name}_difference_maps.png")

    return {
        "sample_name":    sample_name,
        "mid_slice":      mid_slice,
        "masks":          masks,
        "pore_fractions": pore_fracs,
        "preprocessed":   preprocessed,
    }


# =============================================================================
# TASK 3 — Porosity
# =============================================================================

def task3(sample_name: str, masks: dict, mid_slice: str) -> dict:
    summaries = {m: summarize_slice(mask)  for m, mask in masks.items()}
    props     = {m: pore_properties(mask)  for m, mask in masks.items()}

    # ── Porosity bars ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="#0d0d0d")
    dark_ax(ax)
    bars = ax.bar(
        [m.capitalize() for m in METHODS],
        [summaries[m]["porosity"] * 100 for m in METHODS],
        color=[COLORS[m] for m in METHODS], width=0.5, alpha=0.85
    )
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{bar.get_height():.3f}%",
                ha="center", color="white", fontsize=10)
    ax.set_title(f"Porosity per Method — {sample_name}",
                 color="white", fontsize=13)
    ax.set_ylabel("Porosity (%)", color="white")
    plt.tight_layout()
    save_fig(fig, 3, f"{sample_name}_porosity_bars.png")

    # ── Pore size distribution ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d0d0d")
    fig.suptitle(f"Pore Area Distribution — {sample_name}",
                 color="white", fontsize=13)
    for ax, method in zip(axes, METHODS):
        dark_ax(ax)
        areas = props[method]["areas"]
        if areas.size > 0:
            ax.hist(areas, bins=30, color=COLORS[method],
                    alpha=0.85, edgecolor="none")
            ax.axvline(areas.mean(), color="white", lw=1.2, ls="--",
                       label=f"μ={areas.mean():.1f}px")
            ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No pores", transform=ax.transAxes,
                    ha="center", color="white", fontsize=10)
        ax.set_title(f"{method.capitalize()}  n={areas.size}",
                     color="white", fontsize=10)
        ax.set_xlabel("Area (px²)", color="white")
        ax.set_ylabel("Count",     color="white")
    plt.tight_layout()
    save_fig(fig, 3, f"{sample_name}_pore_size_distribution.png")

    # ── Equivalent diameter ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d0d0d")
    fig.suptitle(f"Equivalent Diameter — {sample_name}",
                 color="white", fontsize=13)
    for ax, method in zip(axes, METHODS):
        dark_ax(ax)
        diams = props[method]["equivalent_diameters_um"]
        if diams.size > 0:
            ax.hist(diams, bins=30, color=COLORS[method],
                    alpha=0.85, edgecolor="none")
            ax.axvline(diams.mean(), color="white", lw=1.2, ls="--",
                       label=f"μ={diams.mean():.2f}µm")
            ax.legend(facecolor="#1a1a1a", labelcolor="white", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No pores", transform=ax.transAxes,
                    ha="center", color="white", fontsize=10)
        ax.set_title(f"{method.capitalize()}  n={diams.size}",
                     color="white", fontsize=10)
        ax.set_xlabel("Equiv. Diameter (µm)", color="white")
        ax.set_ylabel("Count",                color="white")
    plt.tight_layout()
    save_fig(fig, 3, f"{sample_name}_equivalent_diameter.png")

    return {
        "sample_name": sample_name,
        "mid_slice":   mid_slice,
        "summaries":   summaries,
    }


# =============================================================================
# TASK 4 — Patch extraction & augmentation
# =============================================================================

def task4(sample_name: str, preprocessed: np.ndarray,
          mid_slice: str) -> dict:
    slc  = preprocessed.astype(np.float32) / 255.0
    mask = bernsen(preprocessed)

    patch_counts = {}

    for patch_size in [64, 128]:
        stride   = patch_size // 2
        h, w     = slc.shape
        patches  = []

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                p = slc [y:y+patch_size, x:x+patch_size]
                m = mask[y:y+patch_size, x:x+patch_size]
                if p.std() < 0.01:
                    continue
                patches.append({"img": p, "mask": m, "y": y, "x": x})

        patch_counts[patch_size] = len(patches)

        # ── Patch grid ───────────────────────────────────────────────────────
        n    = len(patches)
        idxs = list(np.linspace(0, n-1, min(9, n), dtype=int)) if n > 0 else []

        fig, axes = plt.subplots(3, 3, figsize=(9, 9), facecolor="#0d0d0d")
        fig.suptitle(
            f"Patches {patch_size}×{patch_size}px — {sample_name}\n"
            f"({n} total, 50% overlap)",
            color="white", fontsize=12, y=1.02
        )
        for i, ax in enumerate(axes.ravel()):
            if i < len(idxs):
                p = patches[idxs[i]]
                ax.imshow(p["img"], cmap="gray", vmin=0, vmax=1)
                ax.set_title(f"y={p['y']} x={p['x']}",
                             color="#aaaaaa", fontsize=7)
            ax.axis("off")
        plt.tight_layout()
        save_fig(fig, 4, f"{sample_name}_patches_{patch_size}x{patch_size}.png")

        # ── Patch + mask + overlay ───────────────────────────────────────────
        if n >= 3:
            show_idxs = [n//4, n//2, 3*n//4]
            fig, axes = plt.subplots(3, 3, figsize=(9, 9), facecolor="#0d0d0d")
            fig.suptitle(
                f"Patch / Mask / Overlay — {sample_name} — {patch_size}px",
                color="white", fontsize=12
            )
            for row, idx in enumerate(show_idxs):
                p = patches[idx]
                axes[row, 0].imshow(p["img"],  cmap="gray", vmin=0, vmax=1)
                axes[row, 1].imshow(p["mask"], cmap="gray", vmin=0, vmax=1)
                axes[row, 2].imshow(p["img"],  cmap="gray", vmin=0, vmax=1)
                axes[row, 2].imshow(p["mask"], cmap="Reds", alpha=0.4,
                                    vmin=0, vmax=1)
                for ax in axes[row]:
                    ax.axis("off")
            for title, col in zip(["Preprocessed", "Mask", "Overlay"],
                                   [0, 1, 2]):
                axes[0, col].set_title(title, color="white", fontsize=10)
            plt.tight_layout()
            save_fig(fig, 4,
                     f"{sample_name}_patch_mask_{patch_size}x{patch_size}.png")

    # ── Augmentation validation ──────────────────────────────────────────────
    patches_128 = []
    stride = 64
    h, w   = slc.shape
    for y in range(0, h - 128 + 1, stride):
        for x in range(0, w - 128 + 1, stride):
            p = slc [y:y+128, x:x+128]
            m = mask[y:y+128, x:x+128]
            if p.std() >= 0.01:
                patches_128.append({"img": p, "mask": m})

    if patches_128:
        mid_p = patches_128[len(patches_128) // 2]
        fig, axes = plt.subplots(3, 3, figsize=(10, 10), facecolor="#0d0d0d")
        fig.suptitle(
            f"Augmentation Validation — {sample_name}\n"
            "(original + 8 augmentations)",
            color="white", fontsize=12
        )
        axes.ravel()[0].imshow(mid_p["img"], cmap="gray", vmin=0, vmax=1)
        axes.ravel()[0].set_title("Original", color="#aaaaaa", fontsize=9)
        axes.ravel()[0].axis("off")
        for i in range(1, 9):
            aug_img, _ = apply_augmentation(
                mid_p["img"].copy(), mid_p["mask"].copy()
            )
            axes.ravel()[i].imshow(aug_img, cmap="gray", vmin=0, vmax=1)
            axes.ravel()[i].set_title(f"Aug {i}", color="#aaaaaa", fontsize=9)
            axes.ravel()[i].axis("off")
        plt.tight_layout()
        save_fig(fig, 4, f"{sample_name}_augmentation_validation.png")

    return {
        "sample_name":  sample_name,
        "mid_slice":    mid_slice,
        "patch_counts": patch_counts,
    }


# =============================================================================
# TASK 5 — 3D Visualisation
# =============================================================================

def task5(sample_name: str):

    print(f"      Loading {N_SLICES_3D}-slice sub-volume ...")
    prep_vol, mask_vol = load_subvolume(sample_name)
    print(f"      Volume shape: {prep_vol.shape}")

    # ── Surface render ───────────────────────────────────────────────────────
    smoothed = gaussian_filter(prep_vol, sigma=0.8)
    thresh   = threshold_otsu(smoothed)
    try:
        verts, faces, _, _ = marching_cubes(smoothed, level=thresh)
        fig = plt.figure(figsize=(9, 7), facecolor="#0d0d0d")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0d0d0d")
        ax.plot_trisurf(verts[:, 0], verts[:, 1], verts[:, 2],
                        triangles=faces, color="#aaaaaa",
                        alpha=0.3, linewidth=0)
        ax.set_title(f"Air–Solid Surface — {sample_name}",
                     color="white", fontsize=11)
        ax.set_xlabel("Z", color="white", fontsize=8)
        ax.set_ylabel("Y", color="white", fontsize=8)
        ax.set_zlabel("X", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#333333")
        plt.tight_layout()
        save_fig(fig, 5, f"{sample_name}_surface_render.png")
    except Exception as e:
        warnings.warn(f"Surface render failed for {sample_name}: {e}")

    # ── Defect scatter ───────────────────────────────────────────────────────
    pz, py, px = np.where(mask_vol == 1)
    if len(pz) > 0:
        if len(pz) > MAX_POINTS:
            np.random.seed(RANDOM_SEED)
            idx    = np.random.choice(len(pz), MAX_POINTS, replace=False)
            pz, py, px = pz[idx], py[idx], px[idx]
        fig = plt.figure(figsize=(9, 7), facecolor="#0d0d0d")
        ax  = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#0d0d0d")
        sc  = ax.scatter(px, py, pz, c=pz, cmap="Reds", s=1, alpha=0.6)
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.6)
        cbar.set_label("Slice (Z)", color="white", fontsize=8)
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
        ax.set_title(f"Pore Scatter — {sample_name}\n{len(pz):,} voxels",
                     color="white", fontsize=11)
        ax.set_xlabel("X", color="white", fontsize=8)
        ax.set_ylabel("Y", color="white", fontsize=8)
        ax.set_zlabel("Z", color="white", fontsize=8)
        ax.tick_params(colors="white", labelsize=7)
        for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
            pane.fill = False
            pane.set_edgecolor("#333333")
        plt.tight_layout()
        save_fig(fig, 5, f"{sample_name}_defect_scatter.png")

    # ── Orthographic projections ─────────────────────────────────────────────
    proj_xy = prep_vol.max(axis=0)
    proj_xz = prep_vol.max(axis=1)
    proj_yz = prep_vol.max(axis=2)
    mask_xy = mask_vol.sum(axis=0)
    mask_xz = mask_vol.sum(axis=1)
    mask_yz = mask_vol.sum(axis=2)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#0d0d0d")
    fig.suptitle(f"Orthographic Projections — {sample_name}",
                 color="white", fontsize=13, y=1.01)
    for col, (view, mask_, label) in enumerate(zip(
        [proj_xy, proj_xz, proj_yz],
        [mask_xy, mask_xz, mask_yz],
        ["XY (Top)", "XZ (Front)", "YZ (Side)"]
    )):
        axes[0, col].imshow(view,  cmap="gray")
        axes[0, col].set_title(f"{label} — MIP",   color="white", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].imshow(mask_, cmap="hot")
        axes[1, col].set_title(f"{label} — Pores", color="white", fontsize=10)
        axes[1, col].axis("off")
    plt.tight_layout()
    save_fig(fig, 5, f"{sample_name}_orthographic_projections.png")

    # ── Slice mosaic ─────────────────────────────────────────────────────────
    n      = prep_vol.shape[0]
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4*n_cols, 4*n_rows),
                              facecolor="#0d0d0d")
    axes = np.array(axes).reshape(n_rows, n_cols)
    fig.suptitle(f"Slice Mosaic — {sample_name} ({n} slices)",
                 color="white", fontsize=13, y=1.01)
    for i in range(n_rows * n_cols):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        if i < n:
            ax.imshow(prep_vol[i], cmap="gray", vmin=0, vmax=1)
            ax.imshow(mask_vol[i], cmap="Reds",  alpha=0.35, vmin=0, vmax=1)
            ax.set_title(f"Slice {i}", color="#aaaaaa", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    save_fig(fig, 5, f"{sample_name}_slice_mosaic.png")


# =============================================================================
# Cross-sample figures
# =============================================================================

def cross_sample_snr_table(t1_results: list):
    samples    = [r["sample_name"] for r in t1_results]
    step_names = list(t1_results[0]["snrs"].keys())

    fig, ax = plt.subplots(
        figsize=(12, 1.5 + 0.5 * len(samples)), facecolor="#0d0d0d"
    )
    ax.axis("off")
    fig.suptitle("SNR Summary — All Samples × Preprocessing Steps",
                 color="white", fontsize=13)
    tbl = ax.table(
        cellText=[[f"{r['snrs'][s]:.3f}" for s in step_names]
                  for r in t1_results],
        rowLabels=samples,
        colLabels=step_names,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#444444")
        cell.set_text_props(color="white")
        cell.set_facecolor("#222222" if row == 0 or col == -1 else "#0d0d0d")
    plt.tight_layout()
    save_fig(fig, 1, "all_samples_snr_table.png")


def cross_sample_pore_fraction(t2_results: list):
    samples = [r["sample_name"] for r in t2_results]
    x       = np.arange(len(samples))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d0d0d")
    dark_ax(ax)
    for i, method in enumerate(METHODS):
        fracs = [r["pore_fractions"][method] * 100 for r in t2_results]
        bars  = ax.bar(x + i*width, fracs, width,
                       label=method.capitalize(),
                       color=COLORS[method], alpha=0.85)
        for bar, val in zip(bars, fracs):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}%", ha="center",
                    color="white", fontsize=7)
    ax.set_title("Pore Fraction — All Samples × Methods",
                 color="white", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(samples, color="white", fontsize=9)
    ax.set_ylabel("Pore Fraction (%)", color="white")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    plt.tight_layout()
    save_fig(fig, 2, "all_samples_pore_fraction.png")

    # Segmentation overview grid
    n_samples = len(t2_results)
    fig, axes = plt.subplots(n_samples, 3,
                              figsize=(15, 4 * n_samples),
                              facecolor="#0d0d0d")
    if n_samples == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Segmentation Overview — All Samples × Methods",
                 color="white", fontsize=14, y=1.01)
    for row, result in enumerate(t2_results):
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            ax.imshow(result["masks"][method], cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if row == 0:
                ax.set_title(method.capitalize(), color="white", fontsize=11)
            if col == 0:
                ax.set_ylabel(result["sample_name"], color="white", fontsize=9)
                ax.yaxis.set_visible(True)
                ax.tick_params(left=False, labelleft=True)
    plt.tight_layout()
    save_fig(fig, 2, "all_samples_segmentation_overview.png")


def cross_sample_porosity(t3_results: list):
    samples = [r["sample_name"] for r in t3_results]
    x       = np.arange(len(samples))
    width   = 0.25

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d0d0d")
    dark_ax(ax)
    for i, method in enumerate(METHODS):
        vals = [r["summaries"][method]["porosity"] * 100
                for r in t3_results]
        ax.bar(x + i*width, vals, width,
               label=method.capitalize(),
               color=COLORS[method], alpha=0.85)
    ax.set_title("Porosity — All Samples × Methods",
                 color="white", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(samples, color="white", fontsize=9)
    ax.set_ylabel("Porosity (%)", color="white")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    plt.tight_layout()
    save_fig(fig, 3, "all_samples_porosity_comparison.png")


def cross_sample_patch_counts(t4_results: list):
    samples    = [r["sample_name"] for r in t4_results]
    x          = np.arange(len(samples))
    counts_64  = [r["patch_counts"][64]  for r in t4_results]
    counts_128 = [r["patch_counts"][128] for r in t4_results]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d0d0d")
    dark_ax(ax)
    b1 = ax.bar(x - 0.2, counts_64,  0.35, label="64×64",
                color="#e05c5c", alpha=0.85)
    b2 = ax.bar(x + 0.2, counts_128, 0.35, label="128×128",
                color="#5ca8e0", alpha=0.85)
    for bar, val in zip(list(b1)+list(b2), counts_64+counts_128):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+2,
                f"{val:,}", ha="center", color="white", fontsize=8)
    ax.set_title("Patch Count per Sample",
                 color="white", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, color="white", fontsize=9)
    ax.set_ylabel("Number of Patches", color="white")
    ax.legend(facecolor="#1a1a1a", labelcolor="white")
    plt.tight_layout()
    save_fig(fig, 4, "all_samples_patch_count.png")


# =============================================================================
# CSV exports
# =============================================================================

def save_csv_segmentation(t2_results: list):
    rows = []
    for r in t2_results:
        for method in METHODS:
            rows.append({
                "sample":        r["sample_name"],
                "slice":         r["mid_slice"],
                "method":        method,
                "pore_fraction": round(r["pore_fractions"][method], 6),
                "pore_pixels":   int(np.sum(r["masks"][method] == 1)),
                "total_pixels":  r["masks"][method].size,
            })
    _write_csv(out_dir(2) / "segmentation_metrics.csv", rows)


def save_csv_porosity(t3_results: list):
    rows = []
    for r in t3_results:
        for method in METHODS:
            s = r["summaries"][method]
            rows.append({
                "sample":                      r["sample_name"],
                "slice":                       r["mid_slice"],
                "method":                      method,
                "porosity":                    round(s["porosity"], 6),
                "pore_count":                  s["pore_count"],
                "mean_pore_area_px":           round(s["mean_pore_area_px"], 4),
                "mean_pore_area_um2":          round(s["mean_pore_area_um2"], 4),
                "mean_equivalent_diameter_px": round(s["mean_equivalent_diameter_px"], 4),
                "mean_equivalent_diameter_um": round(s["mean_equivalent_diameter_um"], 4),
            })
    _write_csv(out_dir(3) / "porosity_metrics.csv", rows)


def save_csv_patches(t4_results: list):
    rows = []
    for r in t4_results:
        for patch_size in [64, 128]:
            rows.append({
                "sample":      r["sample_name"],
                "slice":       r["mid_slice"],
                "patch_size":  patch_size,
                "patch_count": r["patch_counts"][patch_size],
                "stride":      patch_size // 2,
            })
    _write_csv(out_dir(4) / "patch_stats.csv", rows)


def _write_csv(path, rows: list):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"    [CSV] {path.name}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    create_dirs()
    THESIS_ROOT.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 65)
    print("  XCT Thesis Analysis — All Tasks")
    print("  University West — PODFAM Research Project 2026")
    print("=" * 65)

    # Discover samples
    sample_dirs = sorted(
        d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()
    )

    if not sample_dirs:
        print(f"\nNo samples found in {config.RAW_DATA_DIR}")
        return

    print(f"\n  Found {len(sample_dirs)} sample(s): "
          f"{[d.name for d in sample_dirs]}")
    print(f"  Output root: {THESIS_ROOT}\n")

    t1_results = []
    t2_results = []
    t3_results = []
    t4_results = []

    total_start = time.time()

    for sample_dir in sample_dirs:
        sample_name = sample_dir.name
        print(f"\n{'='*65}")
        print(f"  Sample: {sample_name}")
        print(f"{'='*65}")

        try:
            raw, mid_slice = load_middle_slice(sample_name)
            preprocessed   = preprocess_slice(raw)
            print(f"    Slice: {mid_slice}  Shape: {raw.shape}")

        except Exception as e:
            print(f"    [ERROR] Could not load sample: {e}")
            continue

        # ── Task 1 ──────────────────────────────────────────────────────────
        print(f"\n  [Task 1] Preprocessing analysis ...")
        try:
            t0  = time.time()
            r1  = task1(sample_name, raw, mid_slice)
            t1_results.append(r1)
            print(f"    Done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    [ERROR] Task 1 failed: {e}")

        # ── Task 2 ──────────────────────────────────────────────────────────
        print(f"\n  [Task 2] Segmentation comparison ...")
        try:
            t0  = time.time()
            r2  = task2(sample_name, preprocessed, mid_slice)
            t2_results.append(r2)
            print(f"    Done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    [ERROR] Task 2 failed: {e}")

        # ── Task 3 ──────────────────────────────────────────────────────────
        print(f"\n  [Task 3] Porosity analysis ...")
        try:
            t0  = time.time()
            r3  = task3(sample_name, r2["masks"], mid_slice)
            t3_results.append(r3)
            print(f"    Done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    [ERROR] Task 3 failed: {e}")

        # ── Task 4 ──────────────────────────────────────────────────────────
        print(f"\n  [Task 4] Patch extraction & augmentation ...")
        try:
            t0  = time.time()
            r4  = task4(sample_name, preprocessed, mid_slice)
            t4_results.append(r4)
            print(f"    Done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    [ERROR] Task 4 failed: {e}")

        # ── Task 5 ──────────────────────────────────────────────────────────
        print(f"\n  [Task 5] 3D visualisation ...")
        try:
            t0 = time.time()
            task5(sample_name)
            print(f"    Done in {time.time()-t0:.1f}s")
        except Exception as e:
            print(f"    [ERROR] Task 5 failed: {e}")

    # ── Cross-sample figures & CSVs ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Cross-sample summaries ...")
    print(f"{'='*65}")

    if t1_results:
        cross_sample_snr_table(t1_results)

    if t2_results:
        cross_sample_pore_fraction(t2_results)
        save_csv_segmentation(t2_results)

    if t3_results:
        cross_sample_porosity(t3_results)
        save_csv_porosity(t3_results)

    if t4_results:
        cross_sample_patch_counts(t4_results)
        save_csv_patches(t4_results)

    # ── Final summary ────────────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'='*65}")
    print(f"  ALL TASKS COMPLETE — {elapsed:.1f}s total")
    print(f"  Output root: {THESIS_ROOT}")
    print(f"{'='*65}\n")

    for task_n in range(1, 6):
        task_dir = THESIS_ROOT / f"task{task_n}"
        if task_dir.exists():
            files = sorted(task_dir.glob("*.png")) + \
                    sorted(task_dir.glob("*.csv"))
            print(f"  task{task_n}/  ({len(files)} files)")
            for f in files:
                print(f"    → {f.name}")
    print()


if __name__ == "__main__":
    main()
