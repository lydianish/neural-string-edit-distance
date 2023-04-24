#!/bin/bash

EXPERIMENT_PATH=$SCRATCH/experiments/noise-normalization/neural-string-edit-distance/experiment_022b
TENSORBOARD_DIR=$EXPERIMENT_PATH/models/runs/

tensorboard --logdir=$TENSORBOARD_DIR