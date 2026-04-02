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
parser.add_argument("--radius_f", type=int, default=109)
parser.add_argument("--radius_r", type=int, default=107)
parser.add_argument("--camera_size", type=int, default=2880)
args = parser.parse_args()

data_dir = f"data/{args.scene}"

omni_masks_dir = f"{data_dir}/pre_masking/omni_masks_final"
front_dir = f"{data_dir}/pre_masking/fisheye_masks_camera1"
rear_dir = f"{data_dir}/pre_masking/fisheye_masks_camera2"

os.makedirs(front_dir, exist_ok=True)
os.makedirs(rear_dir,  exist_ok=True)

with open(args.mapping, "r") as f:
    lines = f.readlines()

params_f = eval(lines[0].split('params=')[1].split('(')[0].strip())
params_r = eval(lines[1].split('params=')[1].split('(')[0].strip())

rvec_f = eval(lines[2].strip(), {'array': np.array})
rvec_r = eval(lines[3].strip(), {'array': np.array})

camera_f = pycolmap.Camera(camera_id=0, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=args.camera_size, height=args.camera_size, params=params_f)
camera_r = pycolmap.Camera(camera_id=1, model=pycolmap.CameraModelId.OPENCV_FISHEYE, width=args.camera_size, height=args.camera_size, params=params_r)
    
R_front, _ = cv2.Rodrigues(np.array(rvec_f))
R_rear , _ = cv2.Rodrigues(np.array(rvec_r))

dirs_f = get_cam_ray_dirs(camera_f).cpu().numpy()
dirs_r = get_cam_ray_dirs(camera_r).cpu().numpy()

W_fi, H_fi = camera_f.width, camera_f.height
W_eq, H_eq = W_fi * 2, H_fi

r, Rpix = radial_mask(W_fi, H_fi)
mask_f = (r < (Rpix - (args.radius_f + 0.5)))
mask_r = (r < (Rpix - (args.radius_r + 0.5)))

xyz_f = (R_front @ dirs_f.T).T
xyz_r = (R_rear  @ dirs_r.T).T

xyz_r = np.clip(xyz_r, -1.0, 1.0)

lat_f, lon_f = xyz_to_latlon(xyz_f)
lon_f = -(lon_f+np.pi)

lat_r, lon_r = xyz_to_latlon(xyz_r)
lon_r = -(lon_r)

u_f, v_f = latlon_to_uv(lat_f, lon_f, W_eq, H_eq)
u_r, v_r = latlon_to_uv(lat_r, lon_r, W_eq, H_eq)

u_f_map = u_f.reshape(H_fi, W_fi)
v_f_map = v_f.reshape(H_fi, W_fi)

u_r_map = u_r.reshape(H_fi, W_fi)
v_r_map = v_r.reshape(H_fi, W_fi)

fisheye_front = np.zeros((H_fi, W_fi, 3), dtype=np.uint8)
fisheye_rear  = np.zeros((H_fi, W_fi, 3), dtype=np.uint8)

omni_paths = sorted(glob.glob(os.path.join(omni_masks_dir, "*.png")))

for p in omni_paths:
    img = cv2.imread(p)

    fisheye_front[mask_f] = img[v_f_map[mask_f], u_f_map[mask_f]]
    fisheye_rear[mask_r]  = img[v_r_map[mask_r], u_r_map[mask_r]]

    base = os.path.basename(p)
    cv2.imwrite(os.path.join(front_dir, base), fisheye_front)
    cv2.imwrite(os.path.join(rear_dir,  base), fisheye_rear)
    
    print(f"{front_dir} saved.")
    print(f"{rear_dir} saved.") 