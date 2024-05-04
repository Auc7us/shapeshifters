# ShapeShifters UW-Madison CS766 Project


## Project Website
Please visit our [project website](https://roettges.github.io/shapeshifter_CS766/) for more information about the project and the experiments conducted. 

## Overview
This is repo is a part of the ShapeShifters UW-Madison CS766 Project.  
This repo contains all the training, visualisations, and evaluation scripts for our experiments "Strength of Changeit3D to Gaussian Noise on Point Clouds" and "Robustness of ChangeIt3D to varying language instructions".  
  
Also checkout our [fork of changeit3D repository](https://github.com/kcmacauley/changeit3d/) that contains parts of the code developed specifically for this project including the notebooks for the experiment "Effectiveness of ChangeIt3D with Part Removal".
This fork contains important modules required to run the scripts on this repo; please clone them into the same parent directory.




## Dependencies

```
conda create -n changeit3d python=3.8
conda activate changeit3d
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
https://github.com/kcmacauley/changeit3d
cd changeit3d
conda develop .
cd ..
git clone https://github.com/Auc7us/shapeshifters.git
```


###  Downloading Data

As mentioned in [our fork](https://github.com/Auc7us/shapeshifters/blob/main/README.md#dependencies), download the data from [`optas/changeit3d`](https://github.com/optas/changeit3d?tab=readme-ov-file#shapetalk-dataset--rocket-). After this step, the directory structure should look like this: 

```
shapesifters
changeit3d
data
```

## Experiments 

### Listener Training (Batch-Size Variation)

To run the experiments for this section run, 
```
cd shapeshifters
bash train_listener_ablation.sh
```

This will train the listener model on the 4 classes `chair`, `mug` , `lamp`, `bottle` with the transformer model and save the results in the `data/trained_listener/` directory.  

### Listener Robustness (Noise Variation)

We provide 3 different noise variants for listener robustness training. To run these experiments, use
```
cd shapeshifters
bash scripts/train_listener_noise_sigma_0.0.sh
bash scripts/train_listener_noise_sigma_0.03.sh
bash scripts/train_listener_noise_sigma_0.3.sh
```

This will similarly create results in the `data/trained_listener/` directory. 

### A. ChangeIt3D Performance (Varying Language Instructions)

This section evaluates the pretrained changeit3d model's performance to varying language instructions. To run the experiments for this section run, 
```
python exp_a_shape_talk_language.py
```

#### To visualize the point clouds generated:
```
python visualizations/viz_exp_a.py
```

### B. ChangeIt3D Performance (Source Noise Variation)

This section evaluates the pretrained changeit3d model's performance to varying amounts of noise added to the source point cloud To run the experiments for this section run, 
```
python exp_b_shape_talk_noisy.py
```


##### To visualize the point clouds generated:

```
python visualizations/viz_exp_b.py
```

### Evaluation (Chamfer Distance)

To evaluate the performance of the models, we use the Chamfer Distance metric. To run the evaluation for the experiments and in this repo, run the following command:
```
python evaluation/evaluate_chamfer_distance.py
```

### Data Set Preparation

To prepare datasets for your own experiments you can utilize our sub-repository `DataSetPrep_PlusVisuals` and update the directories with input pointcloud data from the data file generated in the Downloading Data step. Refer to the README file in that subdirectory for more specifics.