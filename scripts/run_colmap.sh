#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-room1}"
DATAPATH="data/${SCENE}"

## Step 1 - feature extraction
colmap312 feature_extractor \
    --image_path "${DATAPATH}/images" \
    --database_path "${DATAPATH}/database.db" \
    --ImageReader.single_camera_per_folder 1 \
    --ImageReader.camera_model OPENCV_FISHEYE

## Step 2 - exhaustive matching
colmap312 exhaustive_matcher \
    --database_path "${DATAPATH}/database.db"

## Step 3 - sparse reconstruction
mkdir -p "${DATAPATH}/sparse"
colmap312 mapper \
    --image_path "${DATAPATH}/images" \
    --database_path "${DATAPATH}/database.db" \
    --output_path "${DATAPATH}/sparse"