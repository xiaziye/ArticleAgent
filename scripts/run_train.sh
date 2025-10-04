#!/bin/bash
# scripts/run_train.sh

set -e  

# pip install -r requirements.txt

export PYTHONPATH="${PYTHONPATH}:$(pwd)"

python train.py --config config/qwen2.5_finetune.yaml

echo "🎉 Training completed!"
