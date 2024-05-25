#!/bin/bash

# Check if MLLM and device parameters are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model> <device>"
    exit 1
fi

# The first command line argument is the model, the second is the device
model="$1"
device="$2"

CUDA_VISIBLE_DEVICES="$device" /home/maxku/anaconda3/envs/arena/bin/python text_guided_t2v.py --model_name "$model"
