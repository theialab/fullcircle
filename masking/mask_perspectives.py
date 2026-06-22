"""Mask objects in perspective images using SAM 3 text prompts."""

import argparse
import os

import cv2
import numpy as np
from PIL import Image


def detect_sam3(fpath, image_shape, predictor, text_prompts):
    predictor.set_image(fpath)
    results = predictor(text=text_prompts)
    if not results or results[0].masks is None or len(results[0].masks.data) == 0:
        return None
    masks_np = results[0].masks.data.cpu().numpy()
    final_mask = np.any(masks_np > 0.5, axis=0).astype(np.uint8)
    if final_mask.shape != image_shape[:2]:
        final_mask = cv2.resize(
            final_mask,
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    return final_mask


def main():
    parser = argparse.ArgumentParser(
        description="Mask objects in perspective images using SAM 3"
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--sam3-model", default="sam3.pt")
    parser.add_argument("--sam3-text", default="person")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step", type=int, default=1)
    args = parser.parse_args()
    if args.step < 1:
        raise ValueError("--step must be >= 1")

    text_prompts = [t.strip() for t in args.sam3_text.split(",") if t.strip()]
    os.makedirs(args.output_path, exist_ok=True)

    print("Loading SAM 3 model...")
    from ultralytics.models.sam import SAM3SemanticPredictor

    predictor = SAM3SemanticPredictor(
        overrides=dict(
            conf=args.conf,
            task="segment",
            mode="predict",
            model=args.sam3_model,
            half=True,
            save=False,
            verbose=False,
            device=args.device,
        )
    )
    print(f"  text prompts: {text_prompts}")

    camera_folders = sorted(
        f for f in os.listdir(args.input_path) if f.startswith("perspective_camera")
    )
    for camera_folder in camera_folders:
        print(f"Processing {camera_folder}...")
        camera_input_dir = os.path.join(args.input_path, camera_folder)
        camera_output_dir = os.path.join(args.output_path, camera_folder)
        camera_mask_dir = os.path.join(camera_output_dir, "masks")
        os.makedirs(camera_mask_dir, exist_ok=True)

        image_files = sorted(
            f
            for f in os.listdir(camera_input_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        selected_files = image_files[:: args.step]
        print(
            f"  Found {len(image_files)} images, processing {len(selected_files)} "
            f"(every {args.step}th)"
        )

        centers_csv = os.path.join(camera_output_dir, "centers.csv")
        with open(centers_csv, "w") as f:
            f.write("filename,cx,cy\n")

            for i, fname in enumerate(selected_files):
                print(f"  Processing {fname} ({i + 1}/{len(selected_files)})")
                fpath = os.path.join(camera_input_dir, fname)
                try:
                    with Image.open(fpath) as image_meta:
                        image_shape = (image_meta.height, image_meta.width)
                except OSError:
                    print(f"  Warning: Could not load {fpath}")
                    continue

                final_mask = detect_sam3(fpath, image_shape, predictor, text_prompts)
                if final_mask is None:
                    print(f"  No objects detected in {fname}")
                    final_mask = np.zeros(image_shape, dtype=np.uint8)
                    cx, cy = float("nan"), float("nan")
                else:
                    moments = cv2.moments(final_mask, binaryImage=True)
                    if moments["m00"] > 0:
                        cx = moments["m10"] / moments["m00"]
                        cy = moments["m01"] / moments["m00"]
                    else:
                        cx, cy = float("nan"), float("nan")

                f.write(f"{fname},{cx:.3f},{cy:.3f}\n")
                cv2.imwrite(
                    os.path.join(camera_mask_dir, fname),
                    (final_mask * 255).astype(np.uint8),
                )

    print("Processing complete!")
    print(f"Results saved to: {args.output_path}")


if __name__ == "__main__":
    main()
