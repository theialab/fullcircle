#!/usr/bin/env bash
set -euo pipefail

SCENE="${1:-room1}"
DATAROOT="${2:-data/fullcircle}"

DATAPATH="${DATAROOT}/${SCENE}"
PRE="pre_masking/sam3"
PRE_ABS="${DATAPATH}/${PRE}"
PERSPECTIVES="${DATAPATH}/pre_masking/perspectives"
PERSPECTIVE_MASKS="${PRE_ABS}/perspective_masks"
MASKS="masks"
COMMON=(--scene "${SCENE}" --data_root "${DATAROOT}" --pre_masking "${PRE}")

activate_env() { set +u; conda activate "$1"; set -u; }

source "$(conda info --base)/etc/profile.d/conda.sh"
activate_env fullcircle

echo "[1/8] omnidirectionals → 6 cubemap faces"
rm -rf "${PERSPECTIVES}" "${PERSPECTIVE_MASKS}"
python masking/omni2perspectives.py \
    --input_path "${DATAPATH}/omni/images" \
    --output_path "${PERSPECTIVES}" > /dev/null 2>&1

activate_env sam3
echo "[2/8] mask perspectives (SAM 3)"
python masking/mask_perspectives.py \
    --input_path "${PERSPECTIVES}" \
    --output_path "${PERSPECTIVE_MASKS}" \
    --sam3-model "${SAM3_MODEL:-${HOME}/models/sam3/sam3.pt}" > /dev/null 2>&1
activate_env fullcircle

echo "[3/8] perspective masks → omnidirectional masks (primary) & directions"
python masking/perspectives2omni.py \
    --input_path "${PERSPECTIVE_MASKS}" \
    --output_path "${PRE_ABS}/omni_masks_primary" > /dev/null 2>&1

echo "[4/8] omnidirectional directions → synthetic fisheyes"
python masking/omni2synthetic.py "${COMMON[@]}" > /dev/null 2>&1

echo "[5/8] mask synthetic fisheyes (SAMv2 automatic segmentation & tracking)"
python thirdparty/sam-ui/scripts/tracking_gui.py \
    --frames-path "${PRE_ABS}/synthetics" \
    --output-path "${PRE_ABS}/synthetic_masks" \
    --headless > /dev/null 2>&1

echo "[6/8] synthetic fisheye masks → omnidirectional masks (final)"
python masking/synthetic2omni.py "${COMMON[@]}" > /dev/null 2>&1

echo "[7/8] omnidirectional masks → raw fisheye masks"
python masking/omni2fisheyes.py "${COMMON[@]}" > /dev/null 2>&1

echo "[8/8] dilate raw fisheye masks"
python masking/dilate.py "${COMMON[@]}" --masks_dir "${MASKS}" > /dev/null 2>&1

echo "Masks saved in ${DATAPATH}/${MASKS}/"
