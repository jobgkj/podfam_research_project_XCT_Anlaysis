"""
=============================================================================
visualize_3d_xct.py — 3D Interactive XCT Volume Visualiser
=============================================================================

Renders a translucent air-solid surface and defect boundary points
from a preprocessed XCT volume using Plotly.

Output: results/figures/<sample_name>_xct_surface.html

Run from project root:
    python scripts/visualize_3d_xct.py
=============================================================================
"""

from pathlib import Path

import numpy as np
import tifffile as tiff
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, binary_erosion
from skimage.filters import threshold_otsu

import config
from config import create_dirs


# =============================================================================
# Configuration (visualisation only)
# =============================================================================

# Optional downsampling factor (2, 4, or 8 recommended)
# Higher = faster render, lower detail
DOWNSAMPLE = 4

# Maximum boundary / defect points to render in the browser
MAX_POINTS = 50_000

# Random seed for reproducible point sampling
RANDOM_SEED = 42


# =============================================================================
# Utilities
# =============================================================================

def select_sample() -> str:
    """
    Discover available samples and prompt the user to select one.

    Returns
    -------
    str
        Selected sample folder name.
    """
    sample_dirs = sorted(
        d for d in config.RAW_DATA_DIR.iterdir() if d.is_dir()
    )

    if not sample_dirs:
        raise RuntimeError(
            f"No sample directories found in {config.RAW_DATA_DIR}"
        )

    print("\nAvailable samples:")
    print("-" * 40)
    for idx, d in enumerate(sample_dirs, start=1):
        print(f"  [{idx}] {d.name}")
    print("-" * 40)

    while True:
        raw = input("Select a sample to visualise (enter number): ").strip()

        try:
            choice = int(raw)
        except ValueError:
            print("  Invalid input — enter a single number.")
            continue

        if choice < 1 or choice > len(sample_dirs):
            print(f"  Choose between 1 and {len(sample_dirs)}.")
            continue

        return sample_dirs[choice - 1].name


def resolve_input_dir(sample_name: str) -> Path:
    """
    Return the input directory for a sample.
    Prefers data/processed/, falls back to data/raw/.
    """
    processed_dir = config.REPO_ROOT / "data" / "processed" / sample_name
    raw_dir       = config.RAW_DATA_DIR / sample_name

    if processed_dir.exists():
        print(f"  Using preprocessed data: {processed_dir}")
        return processed_dir

    if raw_dir.exists():
        print(f"  Preprocessed data not found — using raw: {raw_dir}")
        return raw_dir

    raise FileNotFoundError(
        f"No data found for sample '{sample_name}'. "
        f"Checked:\n  {processed_dir}\n  {raw_dir}"
    )


def load_volume(folder: Path) -> np.ndarray:
    """
    Load all TIFF slices from a folder into a (Z, Y, X) float32 volume.
    Prints an estimated memory usage warning before loading.
    """
    files = sorted(
        list(folder.glob("*.tif")) + list(folder.glob("*.tiff"))
    )

    if not files:
        raise RuntimeError(f"No TIFF files found in {folder}")

    # Estimate memory before loading
    first  = tiff.imread(files[0])
    est_mb = (len(files) * first.nbytes) / (1024 ** 2)
    print(f"  {len(files)} slices — estimated RAM: ~{est_mb:.0f} MB")

    return np.stack(
        [tiff.imread(f).astype(np.float32) for f in files], axis=0
    )


# =============================================================================
# Main
# =============================================================================

def main():

    create_dirs()

    # ------------------------------------------------------------------
    # Sample selection
    # ------------------------------------------------------------------
    sample_name = select_sample()
    print(f"\nVisualising: {sample_name}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load volume
    # ------------------------------------------------------------------
    input_dir = resolve_input_dir(sample_name)

    print("Loading XCT volume...")
    volume_full = load_volume(input_dir)
    print(f"  Volume shape: {volume_full.shape}  (Z, Y, X)")

    # ------------------------------------------------------------------
    # Smooth + threshold
    # ------------------------------------------------------------------
    print("Smoothing volume...")
    volume_smooth = gaussian_filter(volume_full, sigma=0.5)

    print("Computing Otsu threshold...")
    thresh = threshold_otsu(volume_smooth)
    print(f"  Threshold = {thresh:.2f}")

    # ------------------------------------------------------------------
    # Extract air-solid boundary
    # ------------------------------------------------------------------
    print("Extracting air-solid boundary...")
    air_mask      = volume_smooth < thresh
    boundary_mask = air_mask ^ binary_erosion(air_mask)

    # ------------------------------------------------------------------
    # Downsample — applied consistently to both volume and boundary
    # ------------------------------------------------------------------
    print(f"Downsampling by factor {DOWNSAMPLE}...")
    s = slice(None, None, DOWNSAMPLE)

    volume_ds   = volume_smooth [s, s, s]
    boundary_ds = boundary_mask [s, s, s]

    print(f"  Downsampled shape: {volume_ds.shape}")

    # ------------------------------------------------------------------
    # Build coordinate grids for Isosurface
    # Plotly requires explicit (x, y, z, value) — not a raw 3D array
    # ------------------------------------------------------------------
    zi, yi, xi = np.mgrid[
        0:volume_ds.shape[0],
        0:volume_ds.shape[1],
        0:volume_ds.shape[2],
    ]

    # ------------------------------------------------------------------
    # Extract boundary points for Scatter3d
    # ------------------------------------------------------------------
    bz, by, bx = np.where(boundary_ds)

    if len(bz) > MAX_POINTS:
        np.random.seed(RANDOM_SEED)
        idx        = np.random.choice(len(bz), MAX_POINTS, replace=False)
        bz, by, bx = bz[idx], by[idx], bx[idx]

    print(f"  Boundary points to render: {len(bz):,}")

    # ------------------------------------------------------------------
    # Build Plotly figure
    # ------------------------------------------------------------------
    print("Building 3D visualisation...")

    fig = go.Figure()

    # Translucent air-solid isosurface
    fig.add_trace(go.Isosurface(
        x=xi.flatten(),
        y=yi.flatten(),
        z=zi.flatten(),
        value=volume_ds.flatten(),
        isomin=float(thresh * 0.98),
        isomax=float(thresh * 1.02),
        surface_count=1,
        opacity=0.25,
        colorscale="Gray",
        caps=dict(x_show=False, y_show=False, z_show=False),
        showscale=False,
        name="Air–Solid Interface",
    ))

    # Boundary / defect scatter points
    fig.add_trace(go.Scatter3d(
        x=bx, y=by, z=bz,
        mode="markers",
        marker=dict(
            size=2,
            color=bz,
            colorscale="Reds",
            opacity=0.85,
        ),
        name="Boundary / Defects",
    ))

    # Lighting
    fig.update_traces(
        selector=dict(type="isosurface"),
        lighting=dict(
            ambient=0.5,
            diffuse=0.7,
            specular=0.2,
            roughness=0.9,
        ),
    )

    # Layout
    fig.update_layout(
        title=f"XCT Volume — {sample_name} — Air–Solid Surface + Boundary Points",
        scene=dict(
            bgcolor="black",
            xaxis=dict(showgrid=False, zeroline=False, title="X"),
            yaxis=dict(showgrid=False, zeroline=False, title="Y"),
            zaxis=dict(showgrid=False, zeroline=False, title="Slice (Z)"),
        ),
        paper_bgcolor="black",
        font_color="white",
        legend=dict(bgcolor="black"),
    )

    # ------------------------------------------------------------------
    # Save output
    # ------------------------------------------------------------------
    out_file = config.FIGURES_DIR / f"{sample_name}_xct_surface.html"
    fig.write_html(out_file)

    print(f"\nSaved → {out_file}")
    print("Open the file in a browser to interact with the 3D model.")


if __name__ == "__main__":
    main()
