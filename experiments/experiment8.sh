#!/bin/bash

# Declare an array of string with type
declare -a CAMERAS=("canon_eos_1D_mark3" "canon_eos_600D" "fuji" "nikonD5200" "panasonic" "olympus" "sony" "samsung" )

TRAINED_MODEL_PATH="./output_exp/experiment8/"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py conf_exp/experiment8_step1.json Multidataset cube+_shigehler data/multidataset/cube+_shigehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py conf_exp/experiment8_step2.json data/multidataset/nus.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/cube+_shigehler/kmeans_finalaffine_noconf/0/model_best.pth.tar

# Iterate the string array using for loop
for camera in ${CAMERAS[@]}; do
   CUDA_VISIBLE_DEVICES="$GPU_ID" python3 inference_dataset.py conf_exp/experiment8_step2.json data/nus/splits_multicam/$camera.txt --c $TRAINED_MODEL_PATH/checkpoint/Multidataset/nus/kmeans_finalaffine_noconf_finetune_cube_shigehler_multi/0/model_best.pth.tar --c $TRAINED_MODEL_PATH/checkpoint/Multidataset/nus/kmeans_finalaffine_noconf_finetune_cube_shigehler_multi/1/model_best.pth.tar --c $TRAINED_MODEL_PATH/checkpoint/Multidataset/nus/kmeans_finalaffine_noconf_finetune_cube_shigehler_multi/2/model_best.pth.tar -gpu 0 --outputfolder $TRAINED_MODEL_PATH/$camera/
done
