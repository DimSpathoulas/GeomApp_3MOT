# Probabilistic 3d Multi modal Multi object tracking for autonomous navigation
Part of my master thesis: **Probabilistic 3D Multi-Modal Multi-Object Tracking with Machine Learning and Analytic Collision Risk Calculation for Autonomous Navigation.**
## Overview
This is the main module. A review paper in english is under construction.

## Instructions
### 1. Extract Features
If not already extract the Geometric features from CenterPoint detector and the corresponding Appearance features from each camera. Follow our repo.

### 2. Clone our repo and setup the environment
```bash
cd path_to_your_projects/GEOM_APP_Feature_Tracking
conda create --name feature_tracking python=3.8.19
git clone https://github.com/DimSpathoulas/PROB_3D_MULMOD_MOT.git
conda env create -f environment.yml
conda activate feature_tracking
```

### 3. Calculate covariance matrix
I have hardcoded the results for the variances based on centerpoint detector in ```bash cov_centerpoint.py ```, but you can also run the results with ```bash covariance_matrix/cov_calc_centerpoint.py``` with:
```bash
python cov_calc_centerpoint.py > centerpoint_cov.txt
```
### 3. Run main code
You can run the main code for all 3 learning multi-stages both in train and val mode. Note train mode in stages 1 and 2 will give as outputs both the resulting net .pth file and the .json tracking results ready for evaluation.
To run our results:

NOTE: CURRENTLY THERE IS A BUG ISSUE WITH G1, G3 STAGE

### 3. Dimensionaly Reduction
You can vizualise the manifolds individually using ```bash manifolds.py ``` for multiple instances. This will create a directory named features storing the results in .png format.
You can also run SVD on both geometric with ```bash svd_per_point.py ``` for per point SVD in point cloud features or ```bash svd.py ``` for first flattening and then applying SVD on them and ```bash svd_cam.py``` for appearance features . They will output a .pkl file.
You can use these transformation matrices for training the modules with whatever features dimensions you want (you will need to hardcode the changes in main script for that).

### 3. Get Results
To get the tracking metric results run:
```bash
python evaluate_nuscenes.py --output_dir results tracking_output.json > results/tracking_output.txt
```


## The Final Project dir
```bash
# Path_to_your_projects        
└── GEOM_APP_Feature_Tracking
       ├── main.py <-- main code
       ├── main_2.py <-- main code for running only G1,G4
       ├── mmot_mot_3d.py <-- backbone of main.py
       ├── g1_g4.py <-- backbone code of main_2.py
       ├── manifolds.py <-- vizualise geometric features
       ├── svd.py <-- run SVD on geometric features flattened
       ├── svd_per_point.py <-- run SVD on geometric features per point
       ├── svd_cam.py <-- run SVD on camera features
       ├── all.py <-- vizualize results in one manifold (currently not working)
       ├── svd_umap.py             <-- run svd and then vizualize features with umap
       ├── functions             <-- complementary modules
              ├── inner_funcs.py   <-- functions used for tracking
              ├── outer_funcs.py   <-- groundtruth retrieval and output customization
              ├── Kalman_Filter.py   <-- Linear Kalman Filter
```

python evaluate_nuscenes.py --output_dir results real_train_g3_all_classes_no_norm_855.json > results/real_train_g3_all_classes_no_norm_855.txt


val_conv_layer_025.pkl

val_conv_layer51233_thr057_interpolated

train_conv_layer6455_thr034
val_conv_layer6455_thr034
