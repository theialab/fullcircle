import cv2
import numpy as np
import pycolmap
from lib.cam_utils import *
import csv
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene",       required=True)
parser.add_argument("--data_root",   default="data")
parser.add_argument("--pre_masking", default="pre_masking")
parser.add_argument("--mapping",     default="masking/mapping.txt")
parser.add_argument("--radius_r",      type=int, default=107)
parser.add_argument("--camera_size", type=int, default=2880)
args = parser.parse_args()

data_dir = f"{args.data_root}/{args.scene}"
pm = f"{data_dir}/{args.pre_masking}"
synthetics_dir = f"{pm}/synthetics"
synthetic_masks_dir = f"{pm}/synthetic_masks/masks/0"
R_csv = f"{pm}/synthetics/R.csv"

out_omni_masks_dir = f"{pm}/omni_masks_final"
os.makedirs(out_omni_masks_dir, exist_ok=True)

with open(args.mapping, "r") as f:
    lines = f.readlines()
params_r = eval(lines[1].split('params=')[1].split('(')[0].strip())

camera_r = pycolmap.Camera(
    camera_id=1,
    model=pycolmap.CameraModelId.OPENCV_FISHEYE,
    width=args.camera_size, height=args.camera_size,
    params=params_r
)

W_fi, H_fi = camera_r.width, camera_r.height
W_eq, H_eq = camera_r.width * 2, camera_r.height

u = np.arange(W_eq, dtype=np.float32) + 0.5
v = np.arange(H_eq, dtype=np.float32) + 0.5
uu, vv = np.meshgrid(u, v, indexing="xy")
uv = np.stack([uu.ravel(), vv.ravel()], axis=1)

lat, lon = uv_to_latlon(uv, W_eq, H_eq)
dirs_world = latlon_to_xyz(-lat, -lon)

r, Rpix = radial_mask(W_fi, H_fi)
lens = (r < (Rpix - (args.radius_r + 0.5)))

synthetic_paths = sorted(glob.glob(os.path.join(synthetics_dir, "frame_*.png")))

R_lookup = load_R_lookup(R_csv)

for syn_path in synthetic_paths:
    fe_name = os.path.basename(syn_path)
    if fe_name not in R_lookup:
        continue

    mask_path = os.path.join(synthetic_masks_dir, fe_name)
    if os.path.isfile(mask_path):
        img_r = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img_r = cv2.resize(img_r, (W_fi, H_fi), interpolation=cv2.INTER_NEAREST)
        img_r = ((img_r > 0).astype(np.uint8) * 255)
    else:
        img_r = np.zeros((H_fi, W_fi), dtype=np.uint8)

    R_syn = R_lookup[fe_name]

    dirs_cam = (R_syn.T @ dirs_world.T).T

    forward = dirs_cam[:, 2] > 1e-8
    pts_cam = dirs_cam[forward]

    px_py = camera_r.img_from_cam(pts_cam)

    Mx = np.full((H_eq, W_eq), -1, dtype=np.float32)
    My = np.full((H_eq, W_eq), -1, dtype=np.float32)
    My.ravel()[forward] = px_py[:, 1].astype(np.float32)
    Mx.ravel()[forward] = px_py[:, 0].astype(np.float32)

    valid = (Mx >= 0) & (My >= 0)
    if np.any(valid):
        xi = np.clip(Mx[valid].astype(np.int32), 0, W_fi - 1)
        yi = np.clip(My[valid].astype(np.int32), 0, H_fi - 1)
        ok = np.zeros_like(valid)
        ok[valid] = lens[yi, xi]
        Mx[~ok] = -1
        My[~ok] = -1

    omni_mask = cv2.remap(
        img_r, Mx, My,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    omni_mask = ((omni_mask > 0).astype(np.uint8) * 255)

    out_omni_masks_path = os.path.join(out_omni_masks_dir, f"{fe_name}")
    cv2.imwrite(out_omni_masks_path, omni_mask)

    print(f"{out_omni_masks_path} saved.")
