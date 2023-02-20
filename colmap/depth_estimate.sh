#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
SHOWDIR=$3

/home/lzy/anaconda3/envs/MDE/bin/python read_depth.py $CONFIG $CHECKPOINT --show-dir $SHOWDIR
