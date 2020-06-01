# A Multi-Hypothesis Approach to Color Constancy

This is an implementation of "A Multi-Hypothesis Approach to Color Constancy" (CVPR 2020).

# Required hardware

We tested this on a Nvidia Tesla V100 with 32 GB of memory. You can reduce the batch size in the json of every experiment, but results could be different.

# Set dataset paths

You need to set the dataset path in data/paths.json

# Install required packages

You can use the "Dockerfile" included to make sure all the needed packages are installed. Alternatively, we provide a requirements.txt to install required packages with pip (pip install -r requirements.txt).

# Reproducing paper experiments

In order to run the paper experiments, use "bash ./experiments/experiment1.sh":

| Table                             |   Script         |
| --------------------------------- | ---------------- |
| Table 2: Ours                     | experiment1.sh   |
| Table 2: Ours (pretrained)        | experiment2.sh   |
| Table 3: Ours                     | experiment3.sh   |
| Table 3: Ours (pretrained)        | experiment4.sh   |
| Table 4: OMPD: FFCC               | experiment5.sh   |
| Table 4: MDT: FFCC                | experiment6.sh   |
| Table 4: OMPD: Ours (pretrained)  | experiment7.sh   |
| Table 4: MDT: Ours (pretrained)   | experiment8.sh   |
| Table 5: Ours                     | experiment9.sh   |
| Table 5: Ours (pretrained)        | experiment10.sh  |
| Table 6: all rows                 | experiment11.sh  |
| Table 7: Ours                     | experiment12.sh  |
| Table 8: all rows                 | experiment13.sh  |

If you want to run other experiments, here's the way of using the cross-validation, hold-out and inference scripts:

# Cross-validation training
python3 crossvalidation.py EXPERIMENT.json DATASET.txt --outputfolder /PATH -gpu GPU_ID

Note that if you don't set the -gpu you'll be using CPU.

# Hold-out training
python3 hold_out.py EXPERIMENT.json DATASET_CLASS_NAME SUBDATASET DATASET_SPLIT_FILE.txt --outputfolder /PATH --testfile DATASET_HOLDOUT_SPLIT.txt -gpu GPU_ID

# Inference for a single file list
python3 inference.py EXPERIMENT.json DATASET_CLASS_NAME SUBDATASET DATASET_HOLDOUT_SPLIT.txt ./PATH_CHECKPOINT --outputfolder /PATH -gpu GPU_ID

# Inference for a dataset (all folds)
python3 inference_dataset.py EXPERIMENT.json DATASET.txt ./PATH_CHECKPOINT --outputfolder /PATH -gpu GPU_ID
