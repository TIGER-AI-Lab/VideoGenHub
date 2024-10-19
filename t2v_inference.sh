#!/bin/bash

# Check if model and device parameters are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <model> <device> <fps>"
    exit 1
fi

# The first command line argument is the model, the second is the device, the third is the fps
model="$1"
device="$2"
fps="$3"

CUDA_VISIBLE_DEVICES="$device" python3 ./src/videogen_hub/benchmark/text_guided_t2v.py --model_name "$model" --fps $fps