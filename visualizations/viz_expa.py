import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast
from PIL import Image
import sys
sys.path.append('../changeit3d')
from visualization import visualize_point_clouds_3d_v2

target_directory = "../data/shapetalk_dataset/shapetalk/point_clouds/scaled_to_align_rendering"
result_dir = "../data/generation_results"

exp_type = "language"  
folder_name = os.path.join(result_dir, exp_type)
categories = ["lamp", "chair", "vase", "mug", "bottle"]
exp_category = ["baseline", "modified_lang"]

viz_dir = "../data/visualizations/exp_a_lang"
os.makedirs(viz_dir, exist_ok=True)

for category in categories:
    df_dict = {}  
    point_clouds_dict = {}  

    for exp in exp_category:
        result_file = os.path.join(folder_name, f"pcae_dataset_{category}_{exp}.npy")
        results = np.load(result_file, allow_pickle=True).item()
        csv_path = results['csv_path']
        df = pd.read_csv(csv_path)
        df_dict[exp] = df
        point_clouds_dict[exp] = []

        for i in range(len(df)):
            source_file = os.path.join(target_directory, df.loc[i, 'source_uid'] + ".npz")
            target_file = os.path.join(target_directory, df.loc[i, 'target_uid'] + ".npz")
            transformed_shape = results['recons'][1][i]

            with np.load(source_file) as data:
                source_pc = data['pointcloud']
            with np.load(target_file) as data:
                target_pc = data['pointcloud']
            
            point_clouds_dict[exp].append((source_pc, transformed_shape, target_pc))

    for i in range(len(df_dict['baseline'])):
        combined_images = []
        for exp in exp_category:
            point_clouds = point_clouds_dict[exp][i]
            prompt = " ".join(ast.literal_eval(df_dict[exp].loc[i, 'tokens']))
            
            img = visualize_point_clouds_3d_v2(point_clouds, 
                                               title_lst=["Source", f"{exp} Obtained", "Target"],
                                               vis_axis_order=[0, 2, 1],
                                               fig_title=prompt)
            combined_images.append(img)
        
        total_width = max(img.width for img in combined_images)
        total_height = sum(img.height for img in combined_images)
        new_img = Image.new('RGB', (total_width, total_height))
        
        y_offset = 0
        for img in combined_images:
            new_img.paste(img, (0, y_offset))
            y_offset += img.height

        save_path = os.path.join(viz_dir, f"{category}_index_{i}.png")
        new_img.save(save_path)