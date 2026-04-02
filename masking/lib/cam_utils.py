import cv2
import numpy as np
import pycolmap  
import torch
import csv

## fisheye-to-omni functions
def get_cam_ray_dirs(camera, mask=False, radius=80):
    x = np.arange(camera.width,  dtype=np.float32) + 0.5
    y = np.arange(camera.height, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)

    if mask:
        cx, cy = camera.width / 2.0, camera.height / 2.0
        R = min(camera.width, camera.height) / 2.0
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        valid = r < (R - (radius+.5))
        pix_coords = np.stack([x[valid], y[valid]], axis=-1)
    else:
        pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
        
    ip_coords = camera.cam_from_img(pix_coords)
    ip_coords = np.concatenate([ip_coords, np.ones_like(ip_coords[:, :1])], axis=-1)
    ray_dirs  = ip_coords / np.linalg.norm(ip_coords, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)

def xyz_to_latlon(dirs):
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lat = np.arcsin(-y)
    lon = np.arctan2(x, -z)
    return lat, lon

def latlon_to_uv(lat, lon, W, H):
    u = (1 + (  lon/np.pi)) * W/2
    v = (1 - (2*lat/np.pi)) * H/2
    
    return u.astype(np.int32), v.astype(np.int32)

def radial_mask(W_fi, H_fi):
    cx, cy = W_fi / 2.0, H_fi / 2.0
    R = min(W_fi, H_fi) / 2.0
    x, y = np.meshgrid(np.arange(W_fi, dtype=np.float32) + 0.5,
                       np.arange(H_fi, dtype=np.float32) + 0.5)
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    
    return r, R

## omni-to-fisheye functions
def latlon_to_xyz(lat, lon):
    x = np.cos(lat)*np.sin(lon)
    y = np.sin(lat)
    z = -np.cos(lat)*np.cos(lon)
    return np.stack([x,y,z], axis=1)

def uv_to_latlon(uv, W, H):
    u, v = uv[:,0], uv[:,1]
    lat =  np.pi * (0.5 - (v/H))
    lon = -np.pi * (1.0 - (u/W)*2.0)
    return lat, lon

## synthetic fisheye
def normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-12, None)

def look_at_camZ(center_dir, world_up=np.array([0,1,0], dtype=np.float64)):
    """Return world_from_cam rotation with cam +Z aligned to center_dir."""
    z = normalize(center_dir.reshape(1,3))[0]
    up = world_up.astype(np.float64)
    if abs(np.dot(z, up)) > 0.99:
        up = np.array([1,0,0], dtype=np.float64)
    x = normalize(np.cross(up, z).reshape(1,3))[0]
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return R

def load_R_lookup(csv_path):
    R_map = {}
    with open(csv_path, "r", newline="") as f:
        rdr = csv.reader(f)
        first = next(rdr, None)
        if first and len(first) >= 10:
            fname = first[0].strip()
            R = np.array(list(map(float, first[1:10]))).reshape(3,3)
            R_map[fname] = R
        for row in rdr:
            fname = row[0].strip()
            R = np.array(list(map(float, row[1:10]))).reshape(3,3)
            R_map[fname] = R
    return R_map

def fisheye_theta_from_radius(r, f, k1, k2, k3, k4):
    theta = min(np.pi, r / max(f, 1e-6))
    for _ in range(50):
        t2 = theta*theta
        t4 = t2*t2
        t6 = t4*t2
        t8 = t4*t4
        poly  = 1 + k1*t2 + k2*t4 + k3*t6 + k4*t8
        r_d   = f * theta * poly
        if abs(r_d - r) < 1e-9:
            break
        # derivative dr_d/dθ
        dpoly = 2*k1*theta + 4*k2*t3 if (t3:=t2*theta) else 0.0
        dpoly+= 6*k3*(t5:=t4*theta) + 8*k4*(t7:=t6*theta)
        drd   = f * (poly + theta * dpoly)
        theta = theta - (r_d - r)/max(drd, 1e-12)
    return theta

def load_rows(csv_path):
    tmp = []
    with open(csv_path, "r", newline="") as f2:
        rdict2 = csv.DictReader(f2)
        for row in rdict2:
            u = float(row["u"]); v = float(row["v"])
            w = float(row.get("area_px") or row.get("weight", 1.0))
            x_px = u * W_eq; y_px = v * H_eq
            tmp.append((x_px, y_px, w))
    return tmp

# dilate functions
def load_as_binary(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    return bin_img.astype(np.uint8)
