#!/bin/bash
python exp_c_listener_training.py --batch-size=128 --weight-decay=1e-4 --model-type='ablation_model_one' --run-name="ablation_model_1_bs_128" --shape-classes="chair,mug,lamp,bottle"
python exp_c_listener_training.py --batch-size=256 --weight-decay=1e-4 --model-type='ablation_model_one' --run-name="ablation_model_1_bs_256" --shape-classes="chair,mug,lamp,bottle"
python exp_c_listener_training.py --batch-size=128 --weight-decay=1e-4 --model-type='ablation_model_two' --run-name="ablation_model_2_bs_128" --shape-classes="chair,mug,lamp,bottle"
python exp_c_listener_training.py --batch-size=256 --weight-decay=1e-4 --model-type='ablation_model_two' --run-name="ablation_model_2_bs_256" --shape-classes="chair,mug,lamp,bottle"