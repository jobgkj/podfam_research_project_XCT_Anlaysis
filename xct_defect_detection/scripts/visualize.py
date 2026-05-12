"""
=============================================================================
3D Interactive XCT Volume Visualiser
— Translucent Air–Solid Surface + Defect Points
=============================================================================
Run from project root:
    python scripts/visualize_3d_xct.py
=============================================================================
"""

from pathlib import Path
import numpy as np
import tifffile as tiff
import plotly.graph_objects as go

from scipy.ndimage import gaussian_filter, binary_erosion
from src.thresholding import otsu
import config


# ---------------------------------------------------------------------------
# CONFIGURATION (visualisation only)
# ---------------------------------------------------------------------------

# Use preprocessed slices (uint8)
VOLUME_DIR = config.REPO_ROOT / "data" / "processed" / config.SAMPLE_NAME

# Optional visualization downsampling (2, 4, or 8 recommended)
DOWNSAMPLE = 4

# Maximum number of boundary / defect points to render
MAX_POINTS = 50_000


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_volume(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("*.tif"))
    if not files:
        raise RuntimeError(f"No TIFF files found in {folder}")
    return np.stack([tiff.imread(f).astype(np.float32) for f in files], axis=0)


# ---------------------------------------------------------------------------
# Load volume
# ---------------------------------------------------------------------------

print("Loading preprocessed XCT volume (full resolution)...")
volume_full = load_volume(VOLUME_DIR)
print(f"  Volume shape: {volume_full.shape}")

# Optional smoothing improves surface stability
volume_smooth = gaussian_filter(volume_full, sigma=0.5)


# ---------------------------------------------------------------------------
# Air–solid thresholding (global, visualization purpose)
# ---------------------------------------------------------------------------

print("Computing global Otsu threshold (visualization only)...")
thresh = np.mean(volume_smooth)
print(f"  Using threshold = {thresh:.2f}")

air_mask = volume_smooth < thresh


# Extract air–solid interface
print("Extracting air–solid boundary...")
boundary_mask = air_mask ^ binary_erosion(air_mask)


# ---------------------------------------------------------------------------
# Downsample for visualization
# ---------------------------------------------------------------------------

print(f"Downsampling by factor {DOWNSAMPLE}...")
volume = volume_smooth[
    ::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE
]
boundary_mask = boundary_mask[
    ::DOWNSAMPLE, ::DOWNSAMPLE, ::DOWNSAMPLE
]

print(f"  Downsampled shape: {volume.shape}")


# ---------------------------------------------------------------------------
# Extract boundary points
# ---------------------------------------------------------------------------

z, y, x = np.where(boundary_mask)

if len(z) > MAX_POINTS:
    idx = np.random.choice(len(z), MAX_POINTS, replace=False)
    z, y, x = z[idx], y[idx], x[idx]

print(f"  Boundary points to render: {len(z):,}")


# ---------------------------------------------------------------------------
# Build Plotly visualization
# ---------------------------------------------------------------------------

print("Building 3D visualization...")

fig = go.Figure()

# ── Translucent air–solid surface ─────────────────────────────────────────
fig.add_trace(go.Isosurface(
    value=volume,
    isomin=thresh * 0.98,
    isomax=thresh * 1.02,
    surface_count=1,
    opacity=0.25,
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
        color=z,
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


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

out_file = config.REPO_ROOT / "results" / "figures" / "xct_surface_with_points.html"
out_file.parent.mkdir(parents=True, exist_ok=True)

fig.write_html(out_file)
print(f"\nSaved → {out_file}")
print("Open the file in a browser to interact with the 3D model.")
