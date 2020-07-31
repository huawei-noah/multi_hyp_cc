#!/bin/bash

# Declare an array of string with type
declare -a KS=("5" "25" "50" "100" "120" "150" "200" )

TRAINED_MODEL_PATH="./output_exp/experiment13/"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py conf_exp/experiment13/experiment13_step1.json Multidataset nus_shigehler data/multidataset/nus_shigehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

# Iterate the string array using for loop
for K in ${KS[@]}; do
  CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py conf_exp/experiment13/experiment13_step2_k${K}.json Cube plus data/cube/plus.txt -gpu 0 --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/nus_shigehler/kmeans_finalaffine_noconf/0/model_best.pth.tar --outputfolder $TRAINED_MODEL_PATH/k${K}/

  CUDA_VISIBLE_DEVICES="$GPU_ID" python3 inference.py conf_exp/experiment13/experiment13_step2_k${K}.json Cube challenge data/cube/challenge.txt $TRAINED_MODEL_PATH/k${K}/Cube/plus/kmeans_finalaffine_notrain_k${K}/0/model_best.pth.tar -gpu 0 --outputfolder $TRAINED_MODEL_PATH/k${K}/cube_test/
done
