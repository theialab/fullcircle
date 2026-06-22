import argparse
import logging
from pathlib import Path

try:
    from perspectives2omni_two_cubes import run
except ImportError:
    from masking.perspectives2omni_two_cubes import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reconstruct omni masks from one standard 6-face cubemap."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--pattern", default="frame_*")
    parser.add_argument("--mask_subdir", default="masks")
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--omni_height", type=int, default=None)
    parser.add_argument("--chunk_rows", type=int, default=128)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    args = build_parser().parse_args()
    args.num_cameras = 6
    args.second_cube_axis = (1.0, 1.0, 1.0)
    args.second_cube_angle_deg = 45.0
    run(args)
