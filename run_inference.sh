#!/bin/bash

export PYTHONPATH=$(pwd)

python scripts/inference.py --experiment_path models/ \
  --structures_dirpath dataset.evaluation/structures/ \
  --vector_dirpath dataset.evaluation/velocities_lbm/ \
  --batch_size 24 \
  --output_dir dataset.evaluation/velocities_prediction