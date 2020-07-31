#!/bin/bash

# Declare an array of string with type
declare -a CAMERAS=("canon_eos_1D_mark3" "canon_eos_600D" "fuji" "nikonD5200" "panasonic" "olympus" "sony" "samsung" )

TRAINED_MODEL_PATH="./output_exp/experiment5/"
GPU_ID="0"

# Iterate the string array using for loop
for camera in ${CAMERAS[@]}; do
   CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py conf_exp/experiment5/$camera.json data/nus/splits_multicam/$camera.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH/$camera/
done
