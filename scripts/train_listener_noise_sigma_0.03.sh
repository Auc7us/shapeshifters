#!/bin/bash
python exp_c_listener_training.py \
--batch-size=128 \
--weight-decay=1e-4 \
--run-name="ablation_model_1_sigma03" \
--model-type='ablation_model_one' \
--shapetalk-file="../data/noise_added_to_point_clouds/chair_noiseSigma03_exp1.csv"
