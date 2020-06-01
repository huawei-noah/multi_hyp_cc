#!/bin/bash

TRAINED_MODEL_PATH="./output_exp/experiment10/"
GPU_ID="0"

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py conf_exp/experiment10_step1.json Multidataset nus_shigehler data/multidataset/nus_shigehler/all.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 hold_out.py conf_exp/experiment10_step2.json Cube plus data/cube/plus.txt -gpu 0 --outputfolder $TRAINED_MODEL_PATH  --pretrainedmodel $TRAINED_MODEL_PATH/Multidataset/nus_shigehler/kmeans_finalaffine_noconf/0/model_best.pth.tar

CUDA_VISIBLE_DEVICES="$GPU_ID" python3 inference.py conf_exp/experiment10_step2.json Cube challenge data/cube/challenge.txt $TRAINED_MODEL_PATH/Cube/plus/kmeans_finalaffine_noconf_finetune_nus_shigehler/0/model_best.pth.tar -gpu 0 --outputfolder $TRAINED_MODEL_PATH/cube_test/
