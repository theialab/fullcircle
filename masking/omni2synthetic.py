import cv2
import numpy as np
import pycolmap
from lib.cam_utils import *
import csv
import os
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--scene", required=True)
parser.add_argument("--mapping", default="masking/mapping.txt")
parser.add_argument("--radius_r", type=int, default=107)
parser.add_argument("--camera_size", type=int, default=2880)
args = parser.parse_args()

data_dir = f"data/{args.scene}"
omni_dir = f"{data_dir}/omni/images"
centers_dir = f"{data_dir}/pre_masking/omni_masks_primary/centers/"

R_synthetic = f"{data_dir}/pre_masking/synthetics/R.csv"
out_synthetics_dir = f"{data_dir}/pre_masking/synthetics"
os.makedirs(out_synthetics_dir, exist_ok=True)

## Load camera & image
with open(args.mapping, "r") as f:
    lines = f.readlines()

params_r = eval(lines[1].split('params=')[1].split('(')[0].strip())

camera_r = pycolmap.Camera(
    camera_id=1,
    model=pycolmap.CameraModelId.OPENCV_FISHEYE,
    width=args.camera_size, height=args.camera_size,
    params=params_r
)

W_eq, H_eq = camera_r.width * 2, camera_r.height
img_paths = sorted(glob.glob(os.path.join(omni_dir, "*.png")))

W_fi, H_fi = camera_r.width, camera_r.height

dirs_cam = get_cam_ray_dirs(camera_r, mask=True, radius=args.radius_r).cpu().numpy()

r, Rpix = radial_mask(W_fi, H_fi)
lens_mask = (r < (Rpix - (args.radius_r + 0.5)))

for img_path in img_paths:
    stem = os.path.splitext(os.path.basename(img_path))[0]
    img_omni = cv2.imread(img_path)
    centers_csv = os.path.join(centers_dir, f"{stem}.csv")
    
    rows = []
    with open(centers_csv, "r") as f:
        rdict = csv.DictReader(f)
        for row in rdict:
            u = float(row["u"]); v = float(row["v"])
            w = float(row.get("area_px") or row.get("weight", 1.0))
            x_px = u * W_eq; y_px = v * H_eq
            rows.append((x_px, y_px, w))
            
        if len(rows) == 0:
            i = img_paths.index(img_path)

            found = False
            max_d = max(i, len(img_paths) - 1 - i)
            for d in range(1, max_d + 1):
                for j in (i - d, i + d):
                    if j < 0 or j >= len(img_paths):
                        continue

                    near_stem = os.path.splitext(os.path.basename(img_paths[j]))[0]
                    near_csv = os.path.join(centers_dir, f"{near_stem}.csv")

                    try:
                        tmp = load_rows(near_csv)
                        if len(tmp) > 0:
                            rows = tmp
                            print(f"[reuse] {stem} uses centers from {near_stem} (distance {d})")
                            found = True
                            break
                    except FileNotFoundError:
                        continue

                if found:
                    break

            if not found:
                print(f"[skip] {stem} | no valid centers CSV found before/after")
                continue

    uv_pix  = np.array([[x, y] for x, y, _ in rows], dtype=np.float64)
    weights = np.array([w for _, _, w in rows], dtype=np.float64)
    lats, lons = uv_to_latlon(uv_pix, W_eq, H_eq)
    xyzs = latlon_to_xyz(-lats, -lons)
    if np.all(weights <= 0):
        xyz_avg = np.mean(xyzs, axis=0)
    else:
        xyz_avg = np.average(xyzs, axis=0, weights=weights)
    
    R_syn = look_at_camZ(xyz_avg) 
    
    with open(R_synthetic, "a", newline="") as f:
        rowR = [os.path.basename(img_path)] + list(R_syn.reshape(-1))
        csv.writer(f).writerow(rowR)

    xyz_world = (R_syn @ dirs_cam.T).T
    lat_s, lon_s = xyz_to_latlon(xyz_world)
    u_s, v_s = latlon_to_uv(lat_s, -lon_s, W_eq, H_eq)

    map_x = np.full((H_fi, W_fi), -1, np.float32)
    map_y = np.full((H_fi, W_fi), -1, np.float32)    

    map_x[lens_mask] = u_s.astype(np.float32)
    map_y[lens_mask] = v_s.astype(np.float32)

    synthetic = cv2.remap(
        img_omni, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    out_path = os.path.join(out_synthetics_dir, f"{stem}.png")
    ok = cv2.imwrite(out_path, synthetic)
    print(f"{out_path} saved.")