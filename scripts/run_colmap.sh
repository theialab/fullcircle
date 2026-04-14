#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-room1}"
DATAPATH="data/${SCENE}"

COLMAP="$HOME/colmap-3.12/install/bin/colmap"

## Step 1 - feature extraction
${COLMAP} feature_extractor \
    --image_path "${DATAPATH}/images" \
    --database_path "${DATAPATH}/database.db" \
    --ImageReader.mask_path "${DATAPATH}/masks-colmap" \
    --ImageReader.single_camera_per_folder 1 \
    --ImageReader.camera_model OPENCV_FISHEYE

## Step 2 - exhaustive matching
${COLMAP} exhaustive_matcher \
    --database_path "${DATAPATH}/database.db"

## Step 3 - sparse reconstruction
mkdir -p "${DATAPATH}/sparse"
${COLMAP} mapper \
    --image_path "${DATAPATH}/images" \
    --database_path "${DATAPATH}/database.db" \
    --output_path "${DATAPATH}/sparse"
