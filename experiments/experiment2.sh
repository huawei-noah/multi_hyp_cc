#!/bin/bash

TRAINED_MODEL_PATH="./output_exp/experiment2/"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py conf_exp/experiment2_step1.json Multidataset nus_cube+ data/multidataset/nus_cube+/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 crossvalidation.py conf_exp/experiment2_step2.json data/shi_gehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/nus_cube+/kmeans_finalaffine_noconf/0/model_best.pth.tar
