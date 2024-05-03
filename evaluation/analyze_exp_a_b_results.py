import numpy as np
import os
import pandas as pd
from tabulate import tabulate
import torch
import sys
sys.path.append('../changeit3d')
from generic_metrics import chamfer_dists

# from helper import load_pretrained_pc_ae, read_saved_args, ShapeTalkCustomShapeDataset, generate_notebook_args
torch.cuda.empty_cache()
# Directory 
target_directory = "../data/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering"
source_directory = "../data/100pc"
result_dir = "../data/generation_results"


# Experiment 1
exp_type = "noisy_point_clouds_v3"
folder_name = os.path.join(result_dir, exp_type)
# categories = ["lamp", "bottle", "mug"]
# noise_list = ["0.01", "0.02", "0.03"]
categories = ["lamp", "bottle", "mug", "vase"]
noise_list = ["0.00", "0.003", "0.005", "0.008","0.01", "0.02", "0.025", "0.03", "0.035", "0.04", "0.045", "0.05", "0.055", "0.1", "0.2"]

# Dictionary to store noiseless transformed shapes for each category
noiseless_transformed = {}

table1 = []
b_size = 4  # Batch size for Chamfer distance calculations

for category in categories:
    for noise_value in noise_list:
        result_file = os.path.join(folder_name, f"pcae_dataset_noisy_pc_{noise_value}_{category}.npy")
        results = np.load(result_file, allow_pickle=True).item()
        csv_path = results['csv_path']
        df = pd.read_csv(csv_path)

        transformed_shapes = results['recons'][1]

        if noise_value == "0.00":
            source_files = df['source_uid'].apply(lambda x: os.path.join(target_directory, x + ".npz")).tolist()
            source_pcs = [np.load(file)['pointcloud'] for file in source_files]
            noiseless_transformed[category] = transformed_shapes
        
        else:
            source_files = df['source_uid'].apply(lambda x: os.path.join(source_directory, x )).tolist()
            source_pcs = [np.load(file)['pointcloud'] for file in source_files]

        source_pcs = np.array(source_pcs)
        rand_ids2 = np.random.choice(source_pcs.shape[1], 100, replace=False)
        source_pcs = source_pcs[:, rand_ids2]
        average_chamfer_srctr, _ = chamfer_dists(source_pcs, transformed_shapes, b_size)            
        
        # Compute Chamfer distance between noiseless and current noisy transformed shapes
        noiseless_shapes = noiseless_transformed[category]
        average_chamfer_bltr, _ = chamfer_dists(noiseless_shapes, transformed_shapes, b_size)
        table1.append([category, noise_value, average_chamfer_srctr, average_chamfer_bltr])

print(tabulate(table1, headers=["Category", "Noise Value", "Avg Chamfer Distance(CD) b/w Source(sigma = 0 to 0.2) & Transformed(sigma = 0 to 0.2)", "Avg CD b/w Baseline Transformed(sigma = 0) & Transformed(sigma = 0 to 0.2)"]))

table2 = []

exp_type = "language" # language or noisy_point_clouds 
folder_name = os.path.join(result_dir, exp_type)
categories = ["lamp", "chair", "vase"]
exp_category = ["baseline", "modified_lang"]

for category in categories:
    for exp in exp_category:
        result_file = os.path.join(folder_name, f"pcae_dataset_{category}_{exp}.npy")
        results = np.load(result_file, allow_pickle=True).item()
        csv_path = results['csv_path']
    
        df = pd.read_csv(csv_path)
        target_files = df['target_uid'].apply(lambda x: os.path.join(target_directory, x + ".npz")).tolist()
        target_pcs = [np.load(file)['pointcloud'] for file in target_files]

        transformed_shapes = results['recons'][1]
        b_size = 4  # You can adjust this value based on your memory constraints

        average_chamfer, _ = chamfer_dists(target_pcs, transformed_shapes, b_size)
        table2.append([category, exp, average_chamfer])
 
print(tabulate(table2, headers=["Category", "Experiment", "Average Chamfer Distance"]))



