"""
run_all_samples.py — Process all samples in data/raw/
"""

import config
from pathlib import Path
from src.io import load_and_generate_masks


def main():
    for sample_dir in sorted(config.RAW_DATA_DIR.iterdir()):
        if sample_dir.is_dir():
            sample_name = sample_dir.name
            print(f"Processing sample: {sample_name}")

            load_and_generate_masks(
                repo_root=config.REPO_ROOT,
                sample_name=sample_name,
                bernsen_radius=config.BERNSEN_RADIUS,
                bernsen_dct=config.BERNSEN_DCT,
            )

    print("All samples processed.")


if __name__ == "__main__":
    main()
