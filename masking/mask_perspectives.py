import os
import urllib.request
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

SAM_URLS = {'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'}

def ensure_sam_checkpoint(path, model_type):
    if path and os.path.exists(path):
        return path
    default_path = os.path.join(os.path.expanduser('~'), '.cache', 'sam', f'sam_{model_type}.pth')
    if not os.path.exists(default_path):
        os.makedirs(os.path.dirname(default_path), exist_ok=True)
        url = SAM_URLS[model_type]
        print(f"Downloading SAM checkpoint ({model_type}) to {default_path} ...")
        urllib.request.urlretrieve(url, default_path)
    return default_path

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process images with YOLO and SAM for person detection and masking')
    parser.add_argument('--input_path', required=True, help='Base input directory containing camera folders')
    parser.add_argument('--output_path', required=True, help='Base output directory for masks and masked images')
    parser.add_argument('--yolo-model', default='yolov8s.pt', help='Path to YOLO model (default: yolov8s.pt)')
    parser.add_argument('--sam-checkpoint', default=None, help='Path to SAM checkpoint (auto-downloaded if omitted)')
    parser.add_argument('--sam-model-type', default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='SAM model type (default: vit_h)')
    parser.add_argument('--device', default='cuda', help='Device to use (default: cuda)')
    parser.add_argument('--step', type=int, default=1, help='Process every Nth image (default: 30)')
    
    args = parser.parse_args()
    
    # Use provided arguments
    base_input_dir = args.input_path
    output_base_dir = args.output_path
    yolo_model_path = args.yolo_model
    sam_checkpoint = ensure_sam_checkpoint(args.sam_checkpoint, args.sam_model_type)
    model_type = args.sam_model_type
    device = args.device
    step = args.step

    # Create output directory structure
    os.makedirs(output_base_dir, exist_ok=True)

    # Load models
    print("Loading YOLO model...")
    yolo = YOLO(yolo_model_path)
    print("Loading SAM model...")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Process each camera folder
    camera_folders = [f for f in os.listdir(base_input_dir) if f.startswith('perspective_camera')]
    camera_folders.sort()  # Sort to ensure consistent ordering

    for camera_folder in camera_folders:
        print(f"Processing {camera_folder}...")
        
        camera_input_dir = os.path.join(base_input_dir, camera_folder)
        camera_output_dir = os.path.join(output_base_dir, camera_folder)
        camera_mask_dir = os.path.join(camera_output_dir, "masks")
        camera_masked_dir = os.path.join(camera_output_dir, "masked_images")
        
        # Create output directories for this camera
        os.makedirs(camera_output_dir, exist_ok=True)
        os.makedirs(camera_mask_dir, exist_ok=True)
        os.makedirs(camera_masked_dir, exist_ok=True)
        
        # Get all image files and sort them
        image_files = [f for f in os.listdir(camera_input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        image_files.sort()
        
        # Process every Nth image (indices 0, N, 2N, 3N, ...)
        selected_files = image_files[::step]  # This takes every Nth item starting from index 0
        
        print(f"  Found {len(image_files)} images, processing {len(selected_files)} images (every {step}th)")
        
        for i, fname in enumerate(selected_files):
            print(f"  Processing {fname} ({i+1}/{len(selected_files)})")
            
            fpath = os.path.join(camera_input_dir, fname)
            image = cv2.imread(fpath)
            
            if image is None:
                print(f"  Warning: Could not load {fpath}")
                continue
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # YOLO detection
            results = yolo(image_rgb, stream=True, conf=0.35, iou=0.6, classes=[0], agnostic_nms=False, max_det=20)

            all_boxes = []
            for result in results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                for j in range(len(result.boxes)):
                    cls  = int(result.boxes.cls[j])
                    conf = float(result.boxes.conf[j])
                    if cls != 0 or conf < 0.35:
                        continue
                    box = result.boxes.xyxy[j].cpu().numpy().astype(int)
                    all_boxes.append(box)

            if len(all_boxes) == 0:
                print(f"  No persons detected in {fname}")
                # Still save the original image and an empty mask
                mask_path = os.path.join(camera_mask_dir, fname)
                empty_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
                cv2.imwrite(mask_path, empty_mask)
                
                # # Save original image as masked image
                # out_path = os.path.join(camera_masked_dir, fname)
                # cv2.imwrite(out_path, image)
                continue
            
            # SAM segmentation
            predictor.set_image(image_rgb)
            input_boxes = torch.tensor(all_boxes, device=device)

            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image_rgb.shape[:2])
            masks, _, _ = predictor.predict_torch(
                point_coords=None,
                point_labels=None,
                boxes=transformed_boxes,
                multimask_output=False,
            )

            # Apply all person masks
            final_mask = torch.any(masks.squeeze(1), dim=0).cpu().numpy().astype(np.uint8)
            
            # Center of mass for faces (pixel coords)
            M = cv2.moments(final_mask, binaryImage=True)
            if M["m00"] > 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                center_of_mass = (float(cx), float(cy))
            else:
                center_of_mass = (np.nan, np.nan)

            # Log for later merging across faces
            centers_csv = os.path.join(camera_output_dir, "centers.csv")
            header_needed = (not os.path.exists(centers_csv)) or (i == 0)
            with open(centers_csv, "a") as f:
                if header_needed:
                    f.write("filename,cx,cy\n")
                f.write(f"{fname},{center_of_mass[0]:.3f},{center_of_mass[1]:.3f}\n")
            
            # Save mask image
            mask_path = os.path.join(camera_mask_dir, fname)
            final_mask = (final_mask * 255).astype(np.uint8)
            cv2.imwrite(mask_path, final_mask)
            
            # Apply mask to image
            result_img = image_rgb.copy()
            result_img[final_mask > 0] = 0

            # Save masked image
            out_path = os.path.join(camera_masked_dir, fname)
            cv2.imwrite(out_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    print("Processing complete!")
    print(f"Results saved to: {output_base_dir}")

if __name__ == "__main__":
    main()