import argparse
import cv2
import os

from lib.cam_utils import load_as_binary

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--data_root", default="data")
parser.add_argument("--pre_masking", default="pre_masking")
parser.add_argument("--masks_dir", default="masks")
parser.add_argument(
    "--camera",
    nargs="*",
    default=["camera1", "camera2"],
    help="camera id suffixes: pre_masking/fisheye_masks_<name> (default: camera1 camera2)",
)
parser.add_argument(
    "--iter",
    dest="mask_iters",
    nargs="*",
    type=int,
    default=[1, 5],
    help="dilation strengths / output masks-<iter> subdirs (default: 1 5)",
)
args = parser.parse_args()

data_dir = f"{args.data_root}/{args.scene}"
pm = f"{data_dir}/{args.pre_masking}"
dilate_kernel = 9
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))

for mask_iter in args.mask_iters:
    for camera in args.camera:
        folder = f"{pm}/fisheye_masks_{camera}"
        out_dir = f"{data_dir}/{args.masks_dir}/masks-{mask_iter}/{camera}"
        os.makedirs(out_dir, exist_ok=True)

        for filename in sorted(os.listdir(folder)):
            p = os.path.join(folder, filename)
            m = load_as_binary(p)
            out = cv2.dilate(m, kernel, iterations=mask_iter)

            base = os.path.splitext(os.path.basename(p))[0]
            out_name = base + "_mask.png"
            out_path = os.path.join(out_dir, out_name)
            cv2.imwrite(out_path, out)
            print(f"{out_path} saved.")
