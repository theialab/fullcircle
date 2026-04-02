import argparse
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import PIL.ExifTags
import PIL.Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import pycolmap
from pycolmap import logging

def create_virtual_camera(
    omni_height: int, fov_deg: float = 90
) -> pycolmap.Camera:
    """Create a virtual perspective camera."""
    image_size = int(omni_height * fov_deg / 180)
    focal = image_size / (2 * np.tan(np.deg2rad(fov_deg) / 2))
    return pycolmap.Camera.create(0, "PINHOLE", focal, image_size, image_size)

def get_virtual_camera_rays(camera: pycolmap.Camera) -> np.ndarray:
    size = (camera.width, camera.height)
    y, x = np.indices(size).astype(np.float32)
    xy = np.column_stack([x.ravel(), y.ravel()])
    # The center of the upper left most pixel has coordinate (0.5, 0.5)
    xy += 0.5
    xy_norm = camera.cam_from_img(xy)
    rays = np.concatenate([xy_norm, np.ones_like(xy_norm[:, :1])], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays

def spherical_img_from_cam(image_size, rays_in_cam: np.ndarray) -> np.ndarray:
    """Project rays into a 360 omni (spherical) image."""
    if image_size[0] != image_size[1] * 2:
        raise ValueError("Only 360° omnis are supported.")
    if rays_in_cam.ndim != 2 or rays_in_cam.shape[1] != 3:
        raise ValueError(f"{rays_in_cam.shape=} but expected (N,3).")
    r = rays_in_cam.T
    yaw = np.arctan2(r[0], r[2])
    pitch = -np.arctan2(r[1], np.linalg.norm(r[[0, 2]], axis=0))
    u = (1 + yaw / np.pi) / 2
    v = (1 - pitch * 2 / np.pi) / 2
    return np.stack([u, v], -1) * image_size

def get_virtual_rotations(
    num_steps_yaw: int = 8, pitches_deg: Sequence[float] = (-35.0, 35.0)
) -> Sequence[np.ndarray]:
    """Get the relative rotations of the virtual cameras w.r.t. the omni.
    Modified to generate 16 cameras (8 yaw x 2 pitch).
    """
    # Assuming that the omnis are approximately upright.
    cams_from_omni_r = []
    yaws = np.linspace(0, 360, num_steps_yaw, endpoint=False)
    for pitch_deg in pitches_deg:
        yaw_offset = (360 / num_steps_yaw / 2) if pitch_deg > 0 else 0
        for yaw_deg in yaws + yaw_offset:
            cam_from_omni_r = Rotation.from_euler(
                "XY", [-pitch_deg, -yaw_deg], degrees=True
            ).as_matrix()
            cams_from_omni_r.append(cam_from_omni_r)
    return cams_from_omni_r

def create_omni_rig_config(
    cams_from_omni_rotation: Sequence[np.ndarray], ref_idx: int = 0
) -> pycolmap.RigConfig:
    """Create a RigConfig for the given virtual rotations."""
    rig_cameras = []
    for idx, cam_from_omni_rotation in enumerate(cams_from_omni_rotation):
        if idx == ref_idx:
            cam_from_rig = None
        else:
            cam_from_ref_rotation = (
                cam_from_omni_rotation @ cams_from_omni_rotation[ref_idx].T
            )
            cam_from_rig = pycolmap.Rigid3d(
                pycolmap.Rotation3d(cam_from_ref_rotation), np.zeros(3)
            )
        rig_cameras.append(
            pycolmap.RigConfigCamera(
                ref_sensor=idx == ref_idx,
                image_prefix=f"perspective_camera{idx}/",
                cam_from_rig=cam_from_rig,
            )
        )
    return pycolmap.RigConfig(cameras=rig_cameras)

def render_perspective_images(
    omni_image_names: Sequence[str],
    omni_image_dir: Path,
    output_image_dir: Path,
) -> pycolmap.RigConfig:
    cams_from_omni_rotation = get_virtual_rotations()
    rig_config = create_omni_rig_config(cams_from_omni_rotation)

    # We assign each omni pixel to the virtual camera with the closest center.
    cam_centers_in_omni = np.einsum(
        "nij,i->nj", cams_from_omni_rotation, [0, 0, 1]
    )

    camera = omni_size = rays_in_cam = None
    for omni_name in tqdm(omni_image_names):
        omni_path = omni_image_dir / omni_name
        try:
            omni_image = PIL.Image.open(omni_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {omni_path} as it cannot be read.")
            continue
        omni_exif = omni_image.getexif()
        omni_image = np.asarray(omni_image)
        gpsonly_exif = PIL.Image.Exif()
        gpsonly_exif[PIL.ExifTags.IFD.GPSInfo] = omni_exif.get_ifd(
            PIL.ExifTags.IFD.GPSInfo
        )

        omni_height, omni_width, *_ = omni_image.shape
        if omni_width != omni_height * 2:
            raise ValueError("Only 360° omnis are supported.")

        if camera is None:
            camera = create_virtual_camera(omni_height)
            for rig_camera in rig_config.cameras:
                rig_camera.camera = camera
            omni_size = (omni_width, omni_height)
            rays_in_cam = get_virtual_camera_rays(camera)
        else:
            if (omni_width, omni_height) != omni_size:
                raise ValueError(
                    "omnis of different sizes are not supported."
                )

        for cam_idx, cam_from_omni_r in enumerate(cams_from_omni_rotation):
            rays_in_omni = rays_in_cam @ cam_from_omni_r
            xy_in_omni = spherical_img_from_cam(omni_size, rays_in_omni)
            xy_in_omni = xy_in_omni.reshape(
                camera.width, camera.height, 2
            ).astype(np.float32)
            xy_in_omni -= 0.5  # COLMAP to OpenCV pixel origin.
            image = cv2.remap(
                omni_image,
                *np.moveaxis(xy_in_omni, -1, 0),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )
            # We define a mask such that each pixel of the omni has its
            # features extracted only in a single virtual camera.
            closest_camera = np.argmax(rays_in_omni @ cam_centers_in_omni.T, -1)
            mask = (
                ((closest_camera == cam_idx) * 255)
                .astype(np.uint8)
                .reshape(camera.width, camera.height)
            )

            image_name = rig_config.cameras[cam_idx].image_prefix + omni_name
            mask_name = f"{image_name}.png"

            image_path = output_image_dir / image_name
            image_path.parent.mkdir(exist_ok=True, parents=True)
            PIL.Image.fromarray(image).save(image_path, exif=gpsonly_exif)

    return rig_config

def run(args):
    # Define the input and output paths
    input_image_dir = args.input_path
    output_base_dir = args.output_path
    
    # Create output directories for each camera
    for i in range(16):
        cam_dir = output_base_dir / f"perspective_camera{i}"
        cam_dir.mkdir(exist_ok=True, parents=True)

    # Search for input images
    omni_image_names = []
    for img_path in sorted(input_image_dir.glob("frame_*.png")):
        omni_image_names.append(img_path.name)
    
    logging.info(f"Found {len(omni_image_names)} images in {input_image_dir}.")
    
    if not omni_image_names:
        logging.error(f"No frame_*.png images found in {input_image_dir}")
        return

    # Generate 16 virtual camera views
    cams_from_omni_rotation = get_virtual_rotations()
    
    logging.info(f"Generating {len(cams_from_omni_rotation)} virtual camera views")

    camera = omni_size = rays_in_cam = None
    for omni_name in tqdm(omni_image_names, desc="Processing omni images"):
        omni_path = input_image_dir / omni_name
        try:
            omni_image = PIL.Image.open(omni_path)
        except PIL.Image.UnidentifiedImageError:
            logging.info(f"Skipping file {omni_path} as it cannot be read.")
            continue
        
        omni_image = np.asarray(omni_image)
        omni_height, omni_width, *_ = omni_image.shape
        
        if omni_width != omni_height * 2:
            logging.warning(f"Image {omni_name} is not a 360° omni (width={omni_width}, height={omni_height}). Skipping.")
            continue

        if camera is None:
            camera = create_virtual_camera(omni_height)
            omni_size = (omni_width, omni_height)
            rays_in_cam = get_virtual_camera_rays(camera)
        else:
            if (omni_width, omni_height) != omni_size:
                logging.warning(f"Image {omni_name} has different size. Skipping.")
                continue

        for cam_idx, cam_from_omni_r in enumerate(cams_from_omni_rotation):
            rays_in_omni = rays_in_cam @ cam_from_omni_r
            xy_in_omni = spherical_img_from_cam(omni_size, rays_in_omni)
            xy_in_omni = xy_in_omni.reshape(
                camera.width, camera.height, 2
            ).astype(np.float32)
            xy_in_omni -= 0.5  # COLMAP to OpenCV pixel origin.
            image = cv2.remap(
                omni_image,
                *np.moveaxis(xy_in_omni, -1, 0),
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_WRAP,
            )

            # Save the virtual camera image
            cam_dir = output_base_dir / f"perspective_camera{cam_idx}"
            image_path = cam_dir / omni_name
            PIL.Image.fromarray(image).save(image_path)

    logging.info(f"Successfully processed {len(omni_image_names)} omni images")
    logging.info(f"Generated 16 virtual camera views saved to {output_base_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert 360° omni images to 16 virtual camera views")
    parser.add_argument(
        "--input_path", 
        type=Path, 
        help="Input directory containing omni images"
    )
    parser.add_argument(
        "--output_path", 
        type=Path, 
        help="Output base directory for virtual camera images"
    )
    run(parser.parse_args())