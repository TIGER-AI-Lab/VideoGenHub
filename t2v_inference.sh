#!/bin/bash

# Check if model and device parameters are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <model> <device> <fps> [-p]"
    exit 1
fi

# The first command line argument is the model, the second is the device, the third is the fps
model="$1"
device="$2"
fps="$3"

# Check for the -p flag
parallel=false
if [ "$#" -eq 4 ] && [ "$4" == "-p" ]; then
    parallel=true
fi

# Split the device string into an array
IFS=',' read -r -a gpus <<< "$device"

# Determine the number of GPUs
num_gpus=${#gpus[@]}

if [ "$parallel" = true ] && [ "$num_gpus" -gt 1 ]; then
    # Run the parallel script for each GPU
    for i in "${!gpus[@]}"; do
        CUDA_VISIBLE_DEVICES="${gpus[i]}" python3 ./src/videogen_hub/benchmark/text_guided_t2v_parallel.py --model_name "$model" --fps $fps --total_portions $num_gpus --portion_index $i &
    done
    wait
else
    CUDA_VISIBLE_DEVICES="$device" python3 ./src/videogen_hub/benchmark/text_guided_t2v.py --model_name "$model" --fps $fps
fi