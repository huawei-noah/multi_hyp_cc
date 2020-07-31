#!/bin/bash

# Declare an array of string with type
declare -a KS=("5" "25" "50" "100" "120" "150" "200" )

TRAINED_MODEL_PATH="./output_exp/experiment11/"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py conf_exp/experiment11/experiment11_step1.json Multidataset nus_cube+ data/multidataset/nus_cube+/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

# Iterate the string array using for loop
for K in ${KS[@]}; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py conf_exp/experiment11/experiment11_step2_k$K.json data/shi_gehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH/k$K/ --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/nus_cube+/kmeans_finalaffine_noconf/0/model_best.pth.tar
done
