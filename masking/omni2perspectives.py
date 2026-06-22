import argparse
import logging
from pathlib import Path

import numpy as np
import PIL.Image
from tqdm import tqdm

try:
    from omni2perspective_two_cubes import (
        create_virtual_camera,
        get_virtual_rotations,
        make_contact_sheet,
        make_owner_overlay,
        render_all_perspectives,
        save_image,
    )
except ImportError:
    from masking.omni2perspective_two_cubes import (
        create_virtual_camera,
        get_virtual_rotations,
        make_contact_sheet,
        make_owner_overlay,
        render_all_perspectives,
        save_image,
    )


def load_omni_image(path: Path) -> np.ndarray:
    image = PIL.Image.open(path)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return np.asarray(image)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_path)
    output_dir = Path(args.output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    image_paths = sorted(input_dir.glob(args.pattern))
    if not image_paths:
        raise FileNotFoundError(f"No images matching {args.pattern!r} found in {input_dir}.")

    rotations = get_virtual_rotations()[:6]
    for idx in range(len(rotations)):
        (output_dir / f"perspective_camera{idx}").mkdir(exist_ok=True, parents=True)

    logging.info("Rendering %d images into 6 standard cubemap faces.", len(image_paths))

    camera = None
    omni_size = None
    for image_path in tqdm(image_paths, desc="Processing omni images"):
        omni_image = load_omni_image(image_path)
        omni_height, omni_width = omni_image.shape[:2]
        if omni_width != omni_height * 2:
            logging.warning(
                "Skipping %s because it is not 2:1 equirectangular (%dx%d).",
                image_path,
                omni_width,
                omni_height,
            )
            continue

        if camera is None:
            camera = create_virtual_camera(omni_height, args.fov_deg)
            omni_size = (omni_width, omni_height)
        elif (omni_width, omni_height) != omni_size:
            logging.warning(
                "Skipping %s because its size %dx%d differs from the first omni %dx%d.",
                image_path,
                omni_width,
                omni_height,
                omni_size[0],
                omni_size[1],
            )
            continue

        perspective_images = render_all_perspectives(omni_image, camera, rotations)
        for cam_idx, image in enumerate(perspective_images):
            save_image(output_dir / f"perspective_camera{cam_idx}" / image_path.name, image)

        if args.visualize:
            vis_dir = output_dir / "visualization"
            labels = [f"camera{idx}" for idx in range(len(perspective_images))]
            contact_sheet = make_contact_sheet(
                perspective_images,
                labels,
                tile_size=args.visualize_tile_size,
                columns=args.visualize_columns,
            )
            save_image(vis_dir / f"{image_path.stem}_contact_sheet.png", contact_sheet)

            owner_overlay = make_owner_overlay(
                omni_image,
                rotations,
                output_width=args.visualize_overlay_width,
                alpha=args.visualize_overlay_alpha,
            )
            save_image(vis_dir / f"{image_path.stem}_owner_overlay.png", owner_overlay)

    logging.info("Done. Output saved to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert 2:1 omni images to one standard 6-face cubemap."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--pattern", default="frame_*.png")
    parser.add_argument("--fov_deg", type=float, default=90.0)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--visualize_tile_size", type=int, default=360)
    parser.add_argument("--visualize_columns", type=int, default=3)
    parser.add_argument("--visualize_overlay_width", type=int, default=1440)
    parser.add_argument("--visualize_overlay_alpha", type=float, default=0.38)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    run(build_parser().parse_args())
