#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-room1}"
DATAROOT="${2:-data}"
DATAPATH="${DATAROOT}/${SCENE}"

echo "[1/8] omnidirectionals → 16 perspectives"
python masking/omni2perspective.py \
    --input_path "${DATAPATH}/omni/images" \
    --output_path "${DATAPATH}/pre_masking/perspectives" > /dev/null 2>&1

echo "[2/8] mask perspectives (YOLO + SAM)"
python masking/mask_perspectives.py \
    --input_path "${DATAPATH}/pre_masking/perspectives" \
    --output_path "${DATAPATH}/pre_masking/perspective_masks" > /dev/null 2>&1

echo "[3/8] perspective masks → omnidirectional masks (primary) & directions"
python masking/perspective2omni.py \
    --input_path "${DATAPATH}/pre_masking/perspective_masks" \
    --output_path "${DATAPATH}/pre_masking/omni_masks_primary" > /dev/null 2>&1

echo "[4/8] omnidirectional directions → synthetic fisheyes"
python masking/omni2synthetic.py \
    --scene "${SCENE}" --data_root "${DATAROOT}" > /dev/null 2>&1

echo "[5/8] mask synthetic fisheyes (SAMv2 automatic segmentation & tracking)"
python thirdparty/sam-ui/scripts/tracking_gui.py \
    --frames-path "${DATAPATH}/pre_masking/synthetics" \
    --output-path "${DATAPATH}/pre_masking/synthetic_masks" \
    --headless > /dev/null 2>&1

echo "[6/8] synthetic fisheye masks → omnidirectional masks (final)"
python masking/synthetic2omni.py --scene "${SCENE}" --data_root "${DATAROOT}" > /dev/null 2>&1

echo "[7/8] omnidirectional masks → raw fisheye masks"
python masking/omni2fisheye.py --scene "${SCENE}" --data_root "${DATAROOT}" > /dev/null 2>&1

echo "[8/8] dilate raw fisheye masks"
python masking/dilate.py --scene "${SCENE}" --data_root "${DATAROOT}" > /dev/null 2>&1

echo "Masks saved in ${DATAPATH}/masks/"
