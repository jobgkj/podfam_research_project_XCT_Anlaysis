"""
=============================================================================
3D Interactive XCT Volume Visualiser
– Translucent Air–Solid Surface + Defect Points
=============================================================================
Run from project root:
    python visualize.py
=============================================================================
"""

import os
import glob
import numpy as np
import tifffile as tiff
import plotly.graph_objects as go

from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter, binary_erosion

# ── Config ────────────────────────────────────────────────────────────────
VOLUME_DIR = r"data\tiff_output"   # preprocessed TIFF slices
DOWNSAMPLE = 4                     # for visualization ONLY (2, 4, or 8)
MAX_POINTS = 50_000                # max defect points to render

# ── Utilities ─────────────────────────────────────────────────────────────
def load_stack(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.tif")))
    if not files:
        raise RuntimeError(f"No TIFF files found in {folder}")
    return np.stack([tiff.imread(f).astype(np.float32) for f in files], axis=0)


# ── Load volume (FULL resolution first) ───────────────────────────────────
print("Loading TIFF stack (full resolution) ...")
volume_full = load_stack(VOLUME_DIR)
print(f"  Volume shape: {volume_full.shape}")

# Optional light smoothing (helps boundary stability)
volume_smooth = gaussian_filter(volume_full, sigma=0.5)

# ── Thresholding (FULL resolution) ────────────────────────────────────────
print("Computing global Otsu threshold ...")
thresh = threshold_otsu(volume_smooth)
print(f"  Otsu threshold = {thresh:.4f}")

# Air mask (XCT: air = darker)
air_mask = volume_smooth < thresh

# Extract only the air–solid INTERFACE
print("Extracting air–solid boundary ...")
boundary_mask = air_mask ^ binary_erosion(air_mask)

# ── Downsample for visualization ──────────────────────────────────────────
print(f"Downsampling by factor {DOWNSAMPLE} for visualization ...")

volume = volume_smooth[
    ::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE
]

boundary_mask = boundary_mask[
    ::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE
]

print(f"  Downsampled volume shape: {volume.shape}")

# ── Extract boundary points ───────────────────────────────────────────────
z, y, x = np.where(boundary_mask)

# Subsample for performance
if len(z) > MAX_POINTS:
    idx = np.random.choice(len(z), MAX_POINTS, replace=False)
    z, y, x = z[idx], y[idx], x[idx]

print(f"  Boundary points to render: {len(z):,}")

# ── Build visualization ───────────────────────────────────────────────────
print("Building 3D visualisation ...")

fig = go.Figure()

# ── Translucent air–solid surface ─────────────────────────────────────────
fig.add_trace(go.Isosurface(
    value=volume,
    isomin=thresh * 0.98,
    isomax=thresh * 1.02,
    surface_count=1,
    opacity=0.25,                        # translucent
    colorscale="Gray",
    caps=dict(x_show=False, y_show=False, z_show=False),
    showscale=False,
    name="Air–Solid Interface"
))

# ── Boundary / defect points ──────────────────────────────────────────────
fig.add_trace(go.Scatter3d(
    x=x, y=y, z=z,
    mode="markers",
    marker=dict(
        size=2,
        color=z,                        # depth colouring
        colorscale="Reds",
        opacity=0.85
    ),
    name="Boundary / Defects"
))

# ── Lighting & layout ─────────────────────────────────────────────────────
fig.update_traces(
    selector=dict(type="isosurface"),
    lighting=dict(
        ambient=0.5,
        diffuse=0.7,
        specular=0.2,
        roughness=0.9
    )
)

fig.update_layout(
    title="XCT Volume — Translucent Air–Solid Surface with Boundary Points",
    scene=dict(
        bgcolor="black",
        xaxis=dict(showgrid=False, zeroline=False, title="X"),
        yaxis=dict(showgrid=False, zeroline=False, title="Y"),
        zaxis=dict(showgrid=False, zeroline=False, title="Slice"),
    ),
    paper_bgcolor="black",
    font_color="white",
    legend=dict(bgcolor="black")
)

# ── Save output ───────────────────────────────────────────────────────────
out_file = "xct_surface_with_points.html"
fig.write_html(out_file)
print(f"\nSaved → {out_file}")
print("Open it in a browser to interact with the 3D model.")
