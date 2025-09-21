#!/bin/bash

export PYTHONPATH="PATH_TO_REPOSITORY"

python scripts/vectorpredictor.py --name EXPERIMENT_NAME \
--structures_dirpath PATH_TO_DIRECTORY_CONTAINING_STRUCTURES_IN_GIF \
--vector_dirpath PATH_TO_DIRECTORY_CONTAINING_LBM_SOLUTIONS \
--output_dir experiments \
--backbone resnet101 \
--lr 0.0005 \
--minimal_porosity 0 \
--max_epochs 500 \
--batch_size 32 \
--gradient_clip 1.0 \
--seed 100 \
--loss_factor_mass 0.00 \
--loss_factor_div 1.0 \
--loss_factor_tortuosity 0.01 \
--p_flip 0.5 \
--roll_ratio_x 0.3 \
--roll_ratio_y 0.3 \
--minimal_porosity 0 \
--loss_factor_periodic 0.1 \
--smooth_mode none \
--loss_factor_margin 0 \
--post_smooth_sigma 0 \
--loss_factor_hessian 0 \
--loss_factor_inside 5.0 \
--final_conv_kernel_size 3
