"""
=============================================================================
Configuration — XCT Defect Detection Pipeline
=============================================================================
All hyperparameters, paths, and settings in one place.
Change values here only; do not hardcode elsewhere.
=============================================================================
"""

from pathlib import Path

# =============================================================================
# Repository root
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent

# =============================================================================
# Sample
# =============================================================================

# Default single sample — used by io.py / run_pipeline.py
SAMPLE_NAME = "sample_01"

# All samples — auto-discovered at import time
SAMPLE_NAMES = sorted([
    d.name for d in (REPO_ROOT / "data" / "raw").iterdir()
    if d.is_dir()
]) if (REPO_ROOT / "data" / "raw").exists() else []

# =============================================================================
# Data directories
# =============================================================================

RAW_DATA_DIR       = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
MASKS_DIR          = REPO_ROOT / "data" / "masks"

# =============================================================================
# Results directories
# =============================================================================

CKPT_DIR    = REPO_ROOT / "artifacts"
FIGURES_DIR = REPO_ROOT / "results" / "figures"
METRICS_DIR = REPO_ROOT / "results" / "metrics"

# =============================================================================
# Directory creation — call explicitly at pipeline startup
# =============================================================================

def create_dirs() -> None:
    """
    Create all required output directories.
    Call once at the top of any pipeline script — not at import time.
    """
    for d in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MASKS_DIR,
        CKPT_DIR,
        FIGURES_DIR,
        METRICS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Preprocessing
# =============================================================================

NORM_LOW_PERCENTILE  = 1
NORM_HIGH_PERCENTILE = 99
NORM_EPS             = 1e-7

MEDIAN_KERNEL_SIZE   = 3

BHC_POLY_DEGREE      = 3

NLM_PATCH_SIZE       = 5
NLM_PATCH_DIST       = 6
NLM_H_FACTOR         = 0.6    # h = NLM_H_FACTOR * estimated_sigma (adaptive)

RING_FILTER_RADIUS   = 15

# =============================================================================
# Thresholding
# =============================================================================

BERNSEN_RADIUS              = 5      # local window radius in pixels (Kim et al. 2017)
BERNSEN_LOW_CONTRAST_THRESH = 128    # fixed threshold for low-contrast regions

# ------------------------------------------------------------------
# Auto-DCT computation (Kim et al. 2017 method)
# ------------------------------------------------------------------
# DCT is computed per image as:
#     DCT = BERNSEN_DCT_STD_MULTIPLIER × mean(local std of solid phase)
#
# The paper measured avg std = 0.847 for Sample 2 after NLM filtering,
# giving DCT = 18 × 0.847 ≈ 15.
# For noisier images the std will be higher → higher DCT → less over-segmentation.
#
# Set BERNSEN_DCT_AUTO = False and adjust BERNSEN_DCT to use a fixed value.
# ------------------------------------------------------------------

BERNSEN_DCT_AUTO           = True   # True = compute per image (recommended)
BERNSEN_DCT_STD_MULTIPLIER = 18     # from Kim et al. 2017 (18× avg std)
BERNSEN_DCT_N_LOCATIONS    = 5      # grid points per axis for std sampling
BERNSEN_DCT_WINDOW_RADIUS  = 5      # local window radius for std measurement
BERNSEN_DCT_MIN            = 5      # minimum DCT floor (prevents under-segmentation)

# Fallback fixed DCT — only used when BERNSEN_DCT_AUTO = False
# Raised from 15 → 30 to reduce over-segmentation on low-porosity samples
BERNSEN_DCT                = 30

# =============================================================================
# Metrics
# =============================================================================

PIXEL_SIZE_UM   = 1.0   # µm per pixel — update with your scanner's voxel size
MIN_DEFECT_SIZE = 5     # minimum pore area in pixels

# =============================================================================
# Pseudo-label generation
# =============================================================================

MORPH_OPEN_SIZE  = 3
MORPH_CLOSE_SIZE = 3

# =============================================================================
# Patch extraction (2D)
# =============================================================================

PATCH_SIZE    = 256
PATCH_STRIDE  = 128
FG_BG_RATIO   = (1, 3)
MIN_FG_PIXELS = 10

# =============================================================================
# Patch extraction (3D)
# =============================================================================

PATCH_SIZE_3D = (16, 128, 128)   # (D, H, W) — must be divisible by 16

# =============================================================================
# Data augmentation
# =============================================================================

AUG_FLIP_PROB        = 0.5
AUG_ROTATE_PROB      = 0.5
AUG_ELASTIC_PROB     = 0.3
AUG_ELASTIC_ALPHA    = 34
AUG_ELASTIC_SIGMA    = 4

AUG_INTENSITY_PROB   = 0.5
AUG_INTENSITY_RANGE  = (0.9, 1.1)

AUG_NOISE_PROB       = 0.5
AUG_NOISE_STD_RANGE  = (0.01, 0.05)

AUG_GAMMA_PROB       = 0.5
AUG_GAMMA_RANGE      = (0.8, 1.2)

# =============================================================================
# Model (shared by 2D and 3D)
# =============================================================================

ENCODER_CHANNELS = [64, 128, 256, 512]
DROPOUT_RATE     = 0.2

# =============================================================================
# Training
# =============================================================================

DEVICE        = "cuda"   # or "cpu"

BATCH_SIZE_2D = 8
BATCH_SIZE_3D = 1

NUM_EPOCHS    = 50
LEARNING_RATE = 4e-5
WEIGHT_DECAY  = 1e-5

VAL_SPLIT     = 0.2
TEST_SPLIT    = 0.1

LOSS_FUNCTION      = "dice_focal"   # "bce", "dice", "focal", "dice_focal"
DICE_FOCAL_LAMBDA  = 0.5
FOCAL_ALPHA        = 0.25
FOCAL_GAMMA        = 2.0

EARLY_STOP_PATIENCE = 10
SCHEDULER_PATIENCE  = 5

# =============================================================================
# Evaluation
# =============================================================================

DICE_THRESHOLD  = 0.5
ACCEPTANCE_DICE = 0.75
ACCEPTANCE_IOU  = 0.60
ACCEPTANCE_REC  = 0.80

# =============================================================================
# MLflow
# =============================================================================

MLFLOW_EXPERIMENT = "XCT_Defect_Detection"
# config.py

# Number of pixels to erode from the detected boundary
SAMPLE_MASK_EROSION_RADIUS = 5   # adjust based on your dataset

# Intensity threshold to classify air vs. sample
SAMPLE_MASK_AIR_THRESHOLD = 30   # typical range: 20–50 for uint8 XCT slices
