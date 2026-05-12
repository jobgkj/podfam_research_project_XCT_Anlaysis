
from src.io import load_and_generate_masks
import config

load_and_generate_masks(
    repo_root=config.REPO_ROOT,
    sample_name=config.SAMPLE_NAME,
    bernsen_radius=config.BERNSEN_RADIUS,
    bernsen_dct=config.BERNSEN_DCT,
)
