#!/bin/bash

# Check if model path parameter is provided
if [ -z "$1" ]; then
    echo "Error: Model path parameter is required"
    echo "Usage: ./run_cold_start.sh <model_path> [load_human_data]"
    echo "Example: ./run_cold_start.sh /path/to/your/model"
    echo "Example: ./run_cold_start.sh /path/to/your/model true"
    exit 1
fi

MODEL_PATH=$1
LOAD_HUMAN_DATA=${2:-false}

echo "Using model path: $MODEL_PATH"
echo "Load human data: $LOAD_HUMAN_DATA"

accelerate launch --config_file zero3.yaml cold_start.py --model_path "$MODEL_PATH" --load_human_data "$LOAD_HUMAN_DATA"
