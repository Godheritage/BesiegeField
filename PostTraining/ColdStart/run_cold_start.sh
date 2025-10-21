#!/bin/bash

# Check if model path parameter is provided
if [ -z "$1" ]; then
    echo "Error: Model path parameter is required"
    echo "Usage: ./run_cold_start.sh <model_path>"
    echo "Example: ./run_cold_start.sh /path/to/your/model"
    exit 1
fi

MODEL_PATH=$1

echo "Using model path: $MODEL_PATH"

accelerate launch --config_file zero3.yaml cold_start.py --model_path "$MODEL_PATH"
